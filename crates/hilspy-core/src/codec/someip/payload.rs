//! SOME/IP payload serialization (AUTOSAR PRS_SOMEIP) of the parameters of a
//! FIBEX service, over the datatypes and codings `afibex` parses.
//!
//! * Integers and floats: big endian unless the utilization says
//!   `IS-HIGH-LOW-BYTE-ORDER` false; the bit length (utilization, else coded
//!   type, else the base type's) must be whole bytes.
//! * Enums: the enum's coding; decoded to the synonym (`decode_choices`),
//!   encoded from a synonym or a number.
//! * Strings: `A_UNICODE2STRING` is UTF-8 (BOM `EF BB BF`, zero terminated) or
//!   UTF-16 (`UTF-16`/`UCS-2`: BOM, big endian, two zero bytes);
//!   `A_ASCIISTRING` is Latin-1, zero terminated when `TERMINATION="ZERO"`.
//!   `LEADING-LENGTH-INFO-TYPE` puts a length field (`LENGTH-FIELD-SIZE`,
//!   default 32 bits) first, `STANDARD-LENGTH-TYPE` pads to the bit length.
//! * Arrays: a dimension whose minimum and maximum size are equal has that
//!   many elements and no length field; any other dimension has a length
//!   field in bytes (`ARRAY-LENGTH-FIELD-SIZE`, default 32 bits). Numeric
//!   arrays decode to typed arrays, others to lists.
//! * Structures: members in position order, a length field first when the
//!   utilization sets `LENGTH-FIELD-SIZE` (decoding then skips the bytes a
//!   newer struct version appends). Typedefs are their single member.
//! * Unions: length field (default 32 bits), type field (`TYPE-FIELD-SIZE`,
//!   default 32 bits; 1 is the first member), the member, padding; a union is
//!   a mapping with exactly one member.
//!
//! Length and type fields are big endian. Byte order and (with
//! `PASS-ON-TO-SUBELEMENTS`) serialization attributes pass to sub-elements.

use std::fmt::Display;

use afibex::fibex::{
    ArrayDimension, BaseDataType, Category, Coding, ComplexDatatypeClass, Datatype, DatatypeType,
    Encoding, Enum, FibexData, HoTermination, Parameter, SerializationAttributes, Utilization,
};

use crate::error::{Error, Result};
use crate::value::{Array, Value};

/// Serializes the parameters of one FIBEX file.
pub(super) struct Serializer<'a> {
    pub fd: &'a FibexData,
    /// Decode enums to their synonyms.
    pub decode_choices: bool,
}

/// What enclosing parameters pass down to their sub-elements.
#[derive(Clone, Copy, Default)]
pub(super) struct Inherited<'a> {
    big_endian: Option<bool>,
    attributes: Option<&'a SerializationAttributes>,
}

/// The serialization settings of one parameter.
#[derive(Clone, Copy)]
struct Settings<'a> {
    util: Option<&'a Utilization>,
    big_endian: bool,
    attributes: Option<&'a SerializationAttributes>,
}

impl<'a> Settings<'a> {
    fn of(param: &'a Parameter, inherited: Inherited<'a>) -> Self {
        let util = param.utilization.as_ref();
        Settings {
            util,
            big_endian: util
                .and_then(|u| u.is_high_low_byte_order)
                .or(inherited.big_endian)
                .unwrap_or(true),
            attributes: util
                .and_then(|u| u.serialization_attributes.as_ref())
                .or(inherited.attributes),
        }
    }

    fn children(&self) -> Inherited<'a> {
        Inherited {
            big_endian: Some(self.big_endian),
            attributes: self
                .attributes
                .filter(|a| a.pass_on_to_subelements == Some(true)),
        }
    }

    fn length_field_size(&self) -> Option<u32> {
        self.attributes.and_then(|a| a.length_field_size)
    }

    fn bit_length(&self) -> Option<u32> {
        self.util.and_then(|u| u.bit_length)
    }
}

/// Reads a payload front to back.
pub(super) struct Reader<'b> {
    data: &'b [u8],
    pos: usize,
}

impl<'b> Reader<'b> {
    pub(super) fn new(data: &'b [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(super) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize, path: &str) -> Result<&'b [u8]> {
        if n > self.remaining() {
            return Err(Error::decode(format!(
                "{path}: truncated payload ({n} bytes needed, {} left)",
                self.remaining()
            )));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// The next `n` bytes as a reader of their own.
    fn split(&mut self, n: usize, path: &str) -> Result<Reader<'b>> {
        Ok(Reader::new(self.take(n, path)?))
    }

    fn uint(&mut self, bytes: usize, big_endian: bool, path: &str) -> Result<u64> {
        let b = self.take(bytes, path)?;
        let fold = |acc: u64, x: &u8| (acc << 8) | u64::from(*x);
        Ok(if big_endian {
            b.iter().fold(0, fold)
        } else {
            b.iter().rev().fold(0, fold)
        })
    }
}

/// A scalar coding's wire form.
enum Scalar {
    Int { bytes: usize, signed: bool },
    F32,
    F64,
    Str,
}

/// The element type of a typed numeric array.
#[derive(Clone, Copy)]
enum Numeric {
    U8,
    U16,
    U32,
    I32,
    I64,
    F32,
    F64,
}

fn unsupported(path: &str, what: impl Display) -> Error {
    Error::NotFound(format!("{path}: unsupported SOME/IP serialization: {what}"))
}

/// Byte size of a length or type field of `bits` (8, 16 or 32).
fn field_bytes(bits: u32, what: &str, path: &str) -> Result<usize> {
    match bits {
        8 | 16 | 32 => Ok(bits as usize / 8),
        _ => Err(unsupported(path, format!("{what} of {bits} bits"))),
    }
}

/// Write `value` into the `bytes`-byte big-endian field at `at`.
fn patch_len(out: &mut [u8], at: usize, bytes: usize, value: usize, path: &str) -> Result<()> {
    if bytes < 8 && (value as u64) >> (8 * bytes) != 0 {
        return Err(Error::encode(format!(
            "{path}: {value} bytes do not fit a {}-bit length field",
            8 * bytes
        )));
    }
    let be = (value as u64).to_be_bytes();
    out[at..at + bytes].copy_from_slice(&be[8 - bytes..]);
    Ok(())
}

fn write_uint(out: &mut Vec<u8>, value: u64, bytes: usize, big_endian: bool) {
    let be = value.to_be_bytes();
    let b = &be[8 - bytes..];
    if big_endian {
        out.extend_from_slice(b);
    } else {
        out.extend(b.iter().rev());
    }
}

fn check_alignment(bits: Option<u32>, path: &str) -> Result<()> {
    match bits {
        None | Some(0) | Some(8) => Ok(()),
        Some(b) => Err(unsupported(path, format!("BIT-ALIGNMENT {b}"))),
    }
}

/// The name of a parameter or member (its position when it has none).
pub(super) fn param_name(p: &Parameter) -> String {
    p.short_name
        .clone()
        .unwrap_or_else(|| p.position.to_string())
}

/// The elements of an array value.
fn elements(v: &Value, path: &str) -> Result<Vec<Value>> {
    fn conv<T: Copy>(v: &[T], f: impl Fn(T) -> Value) -> Vec<Value> {
        v.iter().map(|x| f(*x)).collect()
    }
    Ok(match v {
        Value::List(l) => l.clone(),
        Value::Array(a) => match a {
            Array::U8(v) => conv(v, |x| Value::Int(x.into())),
            Array::U16(v) => conv(v, |x| Value::Int(x.into())),
            Array::U32(v) => conv(v, |x| Value::Int(x.into())),
            Array::I32(v) => conv(v, |x| Value::Int(x.into())),
            Array::I64(v) => conv(v, Value::Int),
            Array::F32(v) => conv(v, |x| Value::Float(x.into())),
            Array::F64(v) => conv(v, Value::Float),
        },
        v => {
            return Err(Error::encode(format!(
                "{path}: an array needs a sequence, not {v}"
            )))
        }
    })
}

/// Decoded numbers as a typed array (a list where one does not fit).
fn pack(kind: Numeric, values: Vec<Value>) -> Value {
    fn ints<T: TryFrom<i64>>(values: &[Value]) -> Option<Vec<T>> {
        values
            .iter()
            .map(|v| match v {
                Value::Int(i) => T::try_from(*i).ok(),
                _ => None,
            })
            .collect()
    }
    fn floats(values: &[Value]) -> Option<Vec<f64>> {
        values.iter().map(Value::as_f64).collect()
    }
    let array = match kind {
        Numeric::U8 => ints(&values).map(Array::U8),
        Numeric::U16 => ints(&values).map(Array::U16),
        Numeric::U32 => ints(&values).map(Array::U32),
        Numeric::I32 => ints(&values).map(Array::I32),
        Numeric::I64 => ints(&values).map(Array::I64),
        Numeric::F32 => {
            floats(&values).map(|v| Array::F32(v.into_iter().map(|x| x as f32).collect()))
        }
        Numeric::F64 => floats(&values).map(Array::F64),
    };
    array.map_or(Value::List(values), Value::Array)
}

/// A decoded integer as a value: `Int`, or `Float` beyond `i64`.
fn int_value(n: i128) -> Value {
    i64::try_from(n).map_or(Value::Float(n as f64), Value::Int)
}

impl<'a> Serializer<'a> {
    fn datatype(&self, id: &str, path: &str) -> Result<&'a Datatype> {
        self.fd
            .elements
            .datatypes_map_by_id
            .get(id)
            .ok_or_else(|| Error::FibexParse(format!("{path}: unknown datatype {id:?}")))
    }

    fn coding(&self, id: &str, path: &str) -> Result<&'a Coding> {
        self.fd
            .pi
            .codings
            .get(id)
            .ok_or_else(|| Error::FibexParse(format!("{path}: unknown coding {id:?}")))
    }

    /// The coding of a common datatype: the utilization's when it has a bit
    /// length (as `asomeip` does), else the datatype's.
    fn common_coding(&self, coding_ref: &str, s: Settings<'a>, path: &str) -> Result<&'a Coding> {
        let over = s
            .util
            .and_then(|u| u.coding_ref.as_deref())
            .and_then(|r| self.fd.pi.codings.get(r))
            .filter(|c| {
                c.coded_type
                    .as_ref()
                    .is_some_and(|t| t.bit_length.is_some())
            });
        match over {
            Some(c) => Ok(c),
            None => self.coding(coding_ref, path),
        }
    }

    fn enum_coding(&self, coding_ref: &str, s: Settings<'a>, path: &str) -> Result<&'a Coding> {
        let r = s
            .util
            .and_then(|u| u.coding_ref.as_deref())
            .unwrap_or(coding_ref);
        self.coding(r, path)
    }

    fn scalar(&self, coding: &Coding, s: Settings<'_>, path: &str) -> Result<Scalar> {
        let ct = coding.coded_type.as_ref().ok_or_else(|| {
            unsupported(path, format!("coding {:?} has no coded type", coding.id))
        })?;
        let base = ct.base_data_type.as_ref().ok_or_else(|| {
            unsupported(
                path,
                format!("coding {:?} has no base data type", coding.id),
            )
        })?;
        let bits = s.bit_length().or(ct.bit_length);
        let int = |natural: u32, signed: bool| -> Result<Scalar> {
            let b = bits.unwrap_or(natural);
            if b == 0 || b % 8 != 0 || b > 64 {
                return Err(unsupported(path, format!("an integer of {b} bits")));
            }
            Ok(Scalar::Int {
                bytes: b as usize / 8,
                signed,
            })
        };
        match base {
            BaseDataType::AUint8 => int(8, false),
            BaseDataType::AUint16 => int(16, false),
            BaseDataType::AUint32 => int(32, false),
            BaseDataType::AUint64 => int(64, false),
            BaseDataType::AInt8 => int(8, true),
            BaseDataType::AInt16 => int(16, true),
            BaseDataType::AInt32 => int(32, true),
            BaseDataType::AInt64 => int(64, true),
            BaseDataType::AFloat32 if matches!(bits, None | Some(32)) => Ok(Scalar::F32),
            BaseDataType::AFloat64 if matches!(bits, None | Some(64)) => Ok(Scalar::F64),
            BaseDataType::AAsciiString | BaseDataType::AUnicode2String => Ok(Scalar::Str),
            BaseDataType::Other if bits.is_some() => int(0, false),
            other => Err(unsupported(
                path,
                format!("{other:?} with bit length {bits:?}"),
            )),
        }
    }

    /// The typed-array element type of a datatype, if it is a plain number.
    fn numeric(&self, dt: &Datatype, s: Settings<'a>, path: &str) -> Result<Option<Numeric>> {
        let coding_ref = match &dt.datatype {
            DatatypeType::Common(coding_ref) => coding_ref,
            DatatypeType::ComplexType(c) if matches!(c.class, ComplexDatatypeClass::Typedef) => {
                return match c.members.as_slice() {
                    [m] if m.array_dimensions.is_empty() => {
                        let inner = self.datatype(&m.datatype_ref, path)?;
                        self.numeric(inner, Settings::of(m, s.children()), path)
                    }
                    _ => Ok(None),
                };
            }
            _ => return Ok(None),
        };
        let coding = self.common_coding(coding_ref, s, path)?;
        Ok(match self.scalar(coding, s, path)? {
            Scalar::F32 => Some(Numeric::F32),
            Scalar::F64 => Some(Numeric::F64),
            Scalar::Int {
                bytes,
                signed: false,
            } => Some(match bytes {
                1 => Numeric::U8,
                2 => Numeric::U16,
                3 | 4 => Numeric::U32,
                _ => Numeric::I64,
            }),
            Scalar::Int {
                bytes,
                signed: true,
            } => Some(if bytes <= 4 {
                Numeric::I32
            } else {
                Numeric::I64
            }),
            Scalar::Str => None,
        })
    }

    // ---- encoding ----

    /// Serialize `value` as parameter `p`.
    pub(super) fn encode_param(
        &self,
        p: &'a Parameter,
        inherited: Inherited<'a>,
        value: &Value,
        path: &str,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let s = Settings::of(p, inherited);
        let dt = self.datatype(&p.datatype_ref, path)?;
        if p.array_dimensions.is_empty() {
            self.encode_datatype(dt, s, value, path, out)
        } else {
            self.encode_array(&p.array_dimensions, dt, s, value, path, out)
        }
    }

    fn encode_array(
        &self,
        dims: &'a [ArrayDimension],
        dt: &'a Datatype,
        s: Settings<'a>,
        value: &Value,
        path: &str,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let dim = &dims[0];
        check_alignment(dim.bit_alignment, path)?;
        let items = elements(value, path)?;
        let fixed = static_size(dim);
        if let Some(n) = fixed {
            if items.len() != n as usize {
                return Err(Error::encode(format!(
                    "{path}: the array has {n} elements, not {}",
                    items.len()
                )));
            }
        }
        if let Some(max) = dim.maximum_size {
            if items.len() > max as usize {
                return Err(Error::encode(format!(
                    "{path}: {} elements, more than the maximum {max}",
                    items.len()
                )));
            }
        }
        let len_bytes = match fixed {
            Some(_) => 0,
            None => self.array_length_bytes(s, path)?,
        };
        let at = out.len();
        out.resize(at + len_bytes, 0);
        for (i, item) in items.iter().enumerate() {
            let sub = format!("{path}[{i}]");
            if dims.len() > 1 {
                self.encode_array(&dims[1..], dt, s, item, &sub, out)?;
            } else {
                self.encode_datatype(dt, s, item, &sub, out)?;
            }
        }
        if len_bytes > 0 {
            let n = out.len() - at - len_bytes;
            patch_len(out, at, len_bytes, n, path)?;
        }
        Ok(())
    }

    fn array_length_bytes(&self, s: Settings<'_>, path: &str) -> Result<usize> {
        let bits = s
            .attributes
            .and_then(|a| a.array_length_field_size)
            .unwrap_or(32);
        field_bytes(bits, "a dynamic array length field", path)
    }

    fn encode_datatype(
        &self,
        dt: &'a Datatype,
        s: Settings<'a>,
        value: &Value,
        path: &str,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        match &dt.datatype {
            DatatypeType::Common(coding_ref) => {
                let coding = self.common_coding(coding_ref, s, path)?;
                self.encode_coding(coding, s, None, value, path, out)
            }
            DatatypeType::EnumType { coding_ref, enums } => {
                let coding = self.enum_coding(coding_ref, s, path)?;
                self.encode_coding(coding, s, Some(enums), value, path, out)
            }
            DatatypeType::ComplexType(c) => match c.class {
                ComplexDatatypeClass::Typedef => {
                    let [member] = c.members.as_slice() else {
                        return Err(unsupported(path, "a typedef without exactly one member"));
                    };
                    self.encode_param(member, s.children(), value, path, out)
                }
                ComplexDatatypeClass::Structure => {
                    let Value::Map(map) = value else {
                        return Err(Error::encode(format!(
                            "{path}: a structure needs a mapping, not {value}"
                        )));
                    };
                    let names: Vec<String> = c.members.iter().map(param_name).collect();
                    if let Some((k, _)) = map.iter().find(|(k, _)| !names.contains(k)) {
                        return Err(Error::NotFound(format!(
                            "{path}: structure {} has no member {k:?} (members: {})",
                            dt.short_name.as_deref().unwrap_or(&dt.id),
                            names.join(", ")
                        )));
                    }
                    let len_bytes = match s.length_field_size() {
                        None | Some(0) => 0,
                        Some(b) => field_bytes(b, "a structure length field", path)?,
                    };
                    let at = out.len();
                    out.resize(at + len_bytes, 0);
                    for (m, name) in c.members.iter().zip(&names) {
                        let v = map
                            .iter()
                            .find(|(k, _)| k == name)
                            .map(|(_, v)| v)
                            .ok_or_else(|| {
                                Error::encode(format!("{path}: missing member {name:?}"))
                            })?;
                        self.encode_param(m, s.children(), v, &format!("{path}.{name}"), out)?;
                    }
                    if len_bytes > 0 {
                        let n = out.len() - at - len_bytes;
                        patch_len(out, at, len_bytes, n, path)?;
                    }
                    Ok(())
                }
                ComplexDatatypeClass::Union => {
                    let (len_bytes, type_bytes) = union_fields(s, path)?;
                    let (name, v) = match value {
                        Value::Map(m) if m.len() == 1 => (&m[0].0, &m[0].1),
                        _ => {
                            return Err(Error::encode(format!(
                            "{path}: a union needs a mapping with exactly one member, not {value}"
                        )))
                        }
                    };
                    let index = c
                        .members
                        .iter()
                        .position(|m| &param_name(m) == name)
                        .ok_or_else(|| {
                            Error::NotFound(format!(
                                "{path}: union {} has no member {name:?} (members: {})",
                                dt.short_name.as_deref().unwrap_or(&dt.id),
                                c.members
                                    .iter()
                                    .map(param_name)
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ))
                        })?;
                    let at = out.len();
                    out.resize(at + len_bytes, 0);
                    write_uint(out, index as u64 + 1, type_bytes, true);
                    let start = out.len();
                    self.encode_param(
                        &c.members[index],
                        s.children(),
                        v,
                        &format!("{path}.{name}"),
                        out,
                    )?;
                    let n = out.len() - start;
                    patch_len(out, at, len_bytes, n, path)
                }
            },
        }
    }

    fn encode_coding(
        &self,
        coding: &Coding,
        s: Settings<'_>,
        enums: Option<&[Enum]>,
        value: &Value,
        path: &str,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        match self.scalar(coding, s, path)? {
            Scalar::Int { bytes, signed } => {
                let n = integer(value, enums, path)?;
                let bits = 8 * bytes as u32;
                let (min, max) = if signed {
                    (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
                } else {
                    (0, (1i128 << bits) - 1)
                };
                if n < min || n > max {
                    return Err(Error::encode(format!(
                        "{path}: {n} is out of range {min}..={max}"
                    )));
                }
                write_uint(out, n as u64, bytes, s.big_endian);
                Ok(())
            }
            Scalar::F32 => {
                let x = float(value, path)? as f32;
                write_uint(out, x.to_bits().into(), 4, s.big_endian);
                Ok(())
            }
            Scalar::F64 => {
                write_uint(out, float(value, path)?.to_bits(), 8, s.big_endian);
                Ok(())
            }
            Scalar::Str => self.encode_string(coding, s, value, path, out),
        }
    }

    fn encode_string(
        &self,
        coding: &Coding,
        s: Settings<'_>,
        value: &Value,
        path: &str,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let Value::Str(text) = value else {
            return Err(Error::encode(format!(
                "{path}: a string needs a str, not {value}"
            )));
        };
        let ct = coding.coded_type.as_ref().expect("checked by scalar()");
        let mut bytes = Vec::with_capacity(text.len() + 4);
        match ct.base_data_type {
            Some(BaseDataType::AUnicode2String) if utf16(ct.encoding.as_ref(), path)? => {
                bytes.extend_from_slice(&[0xFE, 0xFF]);
                for unit in text.encode_utf16() {
                    bytes.extend_from_slice(&unit.to_be_bytes());
                }
                bytes.extend_from_slice(&[0, 0]);
            }
            Some(BaseDataType::AUnicode2String) => {
                bytes.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
                bytes.extend_from_slice(text.as_bytes());
                bytes.push(0);
            }
            _ => {
                for c in text.chars() {
                    let b = u8::try_from(u32::from(c)).map_err(|_| {
                        Error::encode(format!("{path}: {c:?} is not a Latin-1 character"))
                    })?;
                    bytes.push(b);
                }
                if matches!(ct.termination, Some(HoTermination::Zero)) {
                    bytes.push(0);
                }
            }
        }
        match ct.category {
            Category::LeadingLengthInfoType => {
                let len_bytes = string_length_bytes(s, path)?;
                let at = out.len();
                out.resize(at + len_bytes, 0);
                patch_len(out, at, len_bytes, bytes.len(), path)?;
                out.extend_from_slice(&bytes);
            }
            Category::StandardLengthType => {
                let size = fixed_string_size(ct.bit_length, s, path)?;
                if bytes.len() > size {
                    return Err(Error::encode(format!(
                        "{path}: {:?} needs {} bytes, more than the fixed {size}",
                        text,
                        bytes.len()
                    )));
                }
                bytes.resize(size, 0);
                out.extend_from_slice(&bytes);
            }
            ref other => return Err(unsupported(path, format!("a string of category {other:?}"))),
        }
        Ok(())
    }

    // ---- decoding ----

    /// Deserialize parameter `p`.
    pub(super) fn decode_param(
        &self,
        p: &'a Parameter,
        inherited: Inherited<'a>,
        r: &mut Reader<'_>,
        path: &str,
    ) -> Result<Value> {
        let s = Settings::of(p, inherited);
        let dt = self.datatype(&p.datatype_ref, path)?;
        if p.array_dimensions.is_empty() {
            self.decode_datatype(dt, s, r, path)
        } else {
            self.decode_array(&p.array_dimensions, dt, s, r, path)
        }
    }

    fn decode_array(
        &self,
        dims: &'a [ArrayDimension],
        dt: &'a Datatype,
        s: Settings<'a>,
        r: &mut Reader<'_>,
        path: &str,
    ) -> Result<Value> {
        let dim = &dims[0];
        check_alignment(dim.bit_alignment, path)?;
        let element = |r: &mut Reader<'_>, i: usize| {
            let sub = format!("{path}[{i}]");
            if dims.len() > 1 {
                self.decode_array(&dims[1..], dt, s, r, &sub)
            } else {
                self.decode_datatype(dt, s, r, &sub)
            }
        };
        let mut items = Vec::new();
        if let Some(n) = static_size(dim) {
            for i in 0..n as usize {
                items.push(element(r, i)?);
            }
        } else {
            let len_bytes = self.array_length_bytes(s, path)?;
            let len = r.uint(len_bytes, true, path)? as usize;
            let mut sub = r.split(len, path)?;
            while sub.remaining() > 0 {
                let i = items.len();
                items.push(element(&mut sub, i)?);
            }
        }
        if dims.len() > 1 {
            return Ok(Value::List(items));
        }
        Ok(match self.numeric(dt, s, path)? {
            Some(kind) => pack(kind, items),
            None => Value::List(items),
        })
    }

    fn decode_datatype(
        &self,
        dt: &'a Datatype,
        s: Settings<'a>,
        r: &mut Reader<'_>,
        path: &str,
    ) -> Result<Value> {
        match &dt.datatype {
            DatatypeType::Common(coding_ref) => {
                let coding = self.common_coding(coding_ref, s, path)?;
                self.decode_coding(coding, s, None, r, path)
            }
            DatatypeType::EnumType { coding_ref, enums } => {
                let coding = self.enum_coding(coding_ref, s, path)?;
                self.decode_coding(coding, s, Some(enums), r, path)
            }
            DatatypeType::ComplexType(c) => match c.class {
                ComplexDatatypeClass::Typedef => {
                    let [member] = c.members.as_slice() else {
                        return Err(unsupported(path, "a typedef without exactly one member"));
                    };
                    self.decode_param(member, s.children(), r, path)
                }
                ComplexDatatypeClass::Structure => {
                    let members = |r: &mut Reader<'_>| -> Result<Value> {
                        let mut fields = Vec::with_capacity(c.members.len());
                        for m in &c.members {
                            let name = param_name(m);
                            let v =
                                self.decode_param(m, s.children(), r, &format!("{path}.{name}"))?;
                            fields.push((name, v));
                        }
                        Ok(Value::Map(fields))
                    };
                    match s.length_field_size() {
                        None | Some(0) => members(r),
                        Some(b) => {
                            let bytes = field_bytes(b, "a structure length field", path)?;
                            let len = r.uint(bytes, true, path)? as usize;
                            // Bytes beyond the known members (a newer
                            // version of the struct) are skipped.
                            members(&mut r.split(len, path)?)
                        }
                    }
                }
                ComplexDatatypeClass::Union => {
                    let (len_bytes, type_bytes) = union_fields(s, path)?;
                    let len = r.uint(len_bytes, true, path)? as usize;
                    let kind = r.uint(type_bytes, true, path)?;
                    let mut sub = r.split(len, path)?;
                    let member = usize::try_from(kind)
                        .ok()
                        .and_then(|k| k.checked_sub(1))
                        .and_then(|k| c.members.get(k))
                        .ok_or_else(|| {
                            Error::decode(format!(
                                "{path}: union {} has no member of type {kind}",
                                dt.short_name.as_deref().unwrap_or(&dt.id)
                            ))
                        })?;
                    let name = param_name(member);
                    let v = self.decode_param(
                        member,
                        s.children(),
                        &mut sub,
                        &format!("{path}.{name}"),
                    )?;
                    Ok(Value::Map(vec![(name, v)]))
                }
            },
        }
    }

    fn decode_coding(
        &self,
        coding: &Coding,
        s: Settings<'_>,
        enums: Option<&[Enum]>,
        r: &mut Reader<'_>,
        path: &str,
    ) -> Result<Value> {
        Ok(match self.scalar(coding, s, path)? {
            Scalar::Int { bytes, signed } => {
                let raw = r.uint(bytes, s.big_endian, path)?;
                let n = if signed && bytes < 8 {
                    let shift = 64 - 8 * bytes as u32;
                    i128::from(((raw << shift) as i64) >> shift)
                } else if signed {
                    i128::from(raw as i64)
                } else {
                    i128::from(raw)
                };
                let synonym = enums
                    .filter(|_| self.decode_choices)
                    .and_then(|e| e.iter().find(|e| e.value == n))
                    .and_then(|e| e.synonym.clone());
                synonym.map_or_else(|| int_value(n), Value::Str)
            }
            Scalar::F32 => {
                let raw = r.uint(4, s.big_endian, path)? as u32;
                Value::Float(f32::from_bits(raw).into())
            }
            Scalar::F64 => Value::Float(f64::from_bits(r.uint(8, s.big_endian, path)?)),
            Scalar::Str => Value::Str(self.decode_string(coding, s, r, path)?),
        })
    }

    fn decode_string(
        &self,
        coding: &Coding,
        s: Settings<'_>,
        r: &mut Reader<'_>,
        path: &str,
    ) -> Result<String> {
        let ct = coding.coded_type.as_ref().expect("checked by scalar()");
        let raw = match ct.category {
            Category::LeadingLengthInfoType => {
                let len_bytes = string_length_bytes(s, path)?;
                let len = r.uint(len_bytes, true, path)? as usize;
                r.take(len, path)?
            }
            Category::StandardLengthType => {
                r.take(fixed_string_size(ct.bit_length, s, path)?, path)?
            }
            ref other => return Err(unsupported(path, format!("a string of category {other:?}"))),
        };
        let bad = |what: &str| Error::decode(format!("{path}: invalid {what} string"));
        match ct.base_data_type {
            Some(BaseDataType::AUnicode2String) if utf16(ct.encoding.as_ref(), path)? => {
                let (big_endian, body) = match raw {
                    [0xFE, 0xFF, rest @ ..] => (true, rest),
                    [0xFF, 0xFE, rest @ ..] => (false, rest),
                    rest => (true, rest),
                };
                let units: Vec<u16> = body
                    .chunks_exact(2)
                    .map(|c| {
                        if big_endian {
                            u16::from_be_bytes([c[0], c[1]])
                        } else {
                            u16::from_le_bytes([c[0], c[1]])
                        }
                    })
                    .take_while(|u| *u != 0)
                    .collect();
                String::from_utf16(&units).map_err(|_| bad("UTF-16"))
            }
            Some(BaseDataType::AUnicode2String) => {
                let body = raw.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(raw);
                let end = body.iter().position(|b| *b == 0).unwrap_or(body.len());
                String::from_utf8(body[..end].to_vec()).map_err(|_| bad("UTF-8"))
            }
            _ => Ok(raw
                .iter()
                .take_while(|b| **b != 0)
                .map(|b| char::from(*b))
                .collect()),
        }
    }
}

/// The element count of a fixed-size dimension (minimum = maximum).
fn static_size(dim: &ArrayDimension) -> Option<u32> {
    match (dim.minimum_size, dim.maximum_size) {
        (Some(a), Some(b)) if a == b => Some(a),
        _ => None,
    }
}

/// Byte sizes of a union's length and type fields.
fn union_fields(s: Settings<'_>, path: &str) -> Result<(usize, usize)> {
    let len = s.length_field_size().unwrap_or(32);
    let kind = s.attributes.and_then(|a| a.type_field_size).unwrap_or(32);
    Ok((
        field_bytes(len, "a union length field", path)?,
        field_bytes(kind, "a union type field", path)?,
    ))
}

fn string_length_bytes(s: Settings<'_>, path: &str) -> Result<usize> {
    field_bytes(
        s.length_field_size().unwrap_or(32),
        "a string length field",
        path,
    )
}

fn fixed_string_size(coded: Option<u32>, s: Settings<'_>, path: &str) -> Result<usize> {
    match s.bit_length().or(coded) {
        Some(b) if b > 0 && b % 8 == 0 => Ok(b as usize / 8),
        b => Err(unsupported(
            path,
            format!("a fixed-length string of {b:?} bits"),
        )),
    }
}

/// Whether a unicode string is UTF-16 (else UTF-8).
fn utf16(encoding: Option<&Encoding>, path: &str) -> Result<bool> {
    match encoding {
        None | Some(Encoding::Utf8) => Ok(false),
        Some(Encoding::Utf16 | Encoding::Ucs2) => Ok(true),
        Some(other) => Err(unsupported(path, format!("string encoding {other:?}"))),
    }
}

/// An integer field value: a number, or an enum synonym.
fn integer(value: &Value, enums: Option<&[Enum]>, path: &str) -> Result<i128> {
    match value {
        Value::Int(i) => Ok((*i).into()),
        Value::Bool(b) => Ok((*b).into()),
        Value::Float(x) if x.fract() == 0.0 && x.abs() < 1.9e19 => Ok(*x as i128),
        Value::Str(s) => {
            let e =
                enums.ok_or_else(|| Error::encode(format!("{path}: needs a number, not {s:?}")))?;
            e.iter()
                .find(|e| e.synonym.as_deref() == Some(s.as_str()))
                .map(|e| e.value)
                .ok_or_else(|| {
                    let known: Vec<_> = e.iter().filter_map(|e| e.synonym.as_deref()).collect();
                    Error::encode(format!("{path}: {s:?} is not one of {}", known.join(", ")))
                })
        }
        v => Err(Error::encode(format!("{path}: needs an integer, not {v}"))),
    }
}

fn float(value: &Value, path: &str) -> Result<f64> {
    match value {
        Value::Int(_) | Value::Float(_) => Ok(value.as_f64().expect("a number")),
        v => Err(Error::encode(format!("{path}: needs a number, not {v}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use afibex::fibex::MethodIdType;

    /// A FIBEX service whose one method has a parameter per case.
    const XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<fx:FIBEX xmlns:fx="http://www.asam.net/xml/fbx" xmlns:ho="http://www.asam.net/xml" xmlns:service="http://www.asam.net/xml/fbx/services" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" VERSION="4.1.2">
  <fx:ELEMENTS>
    <fx:SERVICE-INTERFACES>
      <fx:SERVICE-INTERFACE xsi:type="service:SERVICE-INTERFACE-TYPE" ID="S">
        <ho:SHORT-NAME>S</ho:SHORT-NAME>
        <fx:SERVICE-IDENTIFIER>1</fx:SERVICE-IDENTIFIER>
        <service:API-VERSION><service:MAJOR>1</service:MAJOR><service:MINOR>0</service:MINOR></service:API-VERSION>
        <service:METHODS>
          <service:METHOD ID="M">
            <ho:SHORT-NAME>M</ho:SHORT-NAME>
            <service:METHOD-IDENTIFIER>1</service:METHOD-IDENTIFIER>
            <service:INPUT-PARAMETERS>
              <service:INPUT-PARAMETER ID="p_text">
                <ho:SHORT-NAME>text</ho:SHORT-NAME>
                <fx:DATATYPE-REF ID-REF="T_Utf16"/>
                <fx:UTILIZATION><fx:SERIALIZATION-ATTRIBUTES><fx:LENGTH-FIELD-SIZE>16</fx:LENGTH-FIELD-SIZE></fx:SERIALIZATION-ATTRIBUTES></fx:UTILIZATION>
                <service:POSITION>0</service:POSITION>
              </service:INPUT-PARAMETER>
              <service:INPUT-PARAMETER ID="p_pair">
                <ho:SHORT-NAME>pair</ho:SHORT-NAME>
                <fx:DATATYPE-REF ID-REF="T_Pair"/>
                <fx:UTILIZATION><fx:SERIALIZATION-ATTRIBUTES><fx:LENGTH-FIELD-SIZE>8</fx:LENGTH-FIELD-SIZE></fx:SERIALIZATION-ATTRIBUTES></fx:UTILIZATION>
                <service:POSITION>1</service:POSITION>
              </service:INPUT-PARAMETER>
              <service:INPUT-PARAMETER ID="p_grid">
                <ho:SHORT-NAME>grid</ho:SHORT-NAME>
                <fx:DATATYPE-REF ID-REF="T_Handle"/>
                <fx:ARRAY-DECLARATION>
                  <fx:ARRAY-DIMENSION><fx:DIMENSION>1</fx:DIMENSION></fx:ARRAY-DIMENSION>
                  <fx:ARRAY-DIMENSION><fx:DIMENSION>2</fx:DIMENSION><fx:MINIMUM-SIZE>2</fx:MINIMUM-SIZE><fx:MAXIMUM-SIZE>2</fx:MAXIMUM-SIZE></fx:ARRAY-DIMENSION>
                </fx:ARRAY-DECLARATION>
                <fx:UTILIZATION><fx:SERIALIZATION-ATTRIBUTES><fx:ARRAY-LENGTH-FIELD-SIZE>8</fx:ARRAY-LENGTH-FIELD-SIZE></fx:SERIALIZATION-ATTRIBUTES></fx:UTILIZATION>
                <service:POSITION>2</service:POSITION>
              </service:INPUT-PARAMETER>
              <service:INPUT-PARAMETER ID="p_odd">
                <ho:SHORT-NAME>odd</ho:SHORT-NAME>
                <fx:DATATYPE-REF ID-REF="T_Int32"/>
                <fx:UTILIZATION><fx:BIT-LENGTH>24</fx:BIT-LENGTH><fx:IS-HIGH-LOW-BYTE-ORDER>false</fx:IS-HIGH-LOW-BYTE-ORDER></fx:UTILIZATION>
                <service:POSITION>3</service:POSITION>
              </service:INPUT-PARAMETER>
            </service:INPUT-PARAMETERS>
          </service:METHOD>
        </service:METHODS>
      </fx:SERVICE-INTERFACE>
    </fx:SERVICE-INTERFACES>
    <fx:DATATYPES>
      <fx:DATATYPE xsi:type="fx:COMMON-DATATYPE-TYPE" ID="T_Utf16"><ho:SHORT-NAME>Utf16</ho:SHORT-NAME><fx:CODING-REF ID-REF="C_Utf16"/></fx:DATATYPE>
      <fx:DATATYPE xsi:type="fx:COMMON-DATATYPE-TYPE" ID="T_UInt8"><ho:SHORT-NAME>UInt8</ho:SHORT-NAME><fx:CODING-REF ID-REF="C_UInt8"/></fx:DATATYPE>
      <fx:DATATYPE xsi:type="fx:COMMON-DATATYPE-TYPE" ID="T_UInt16"><ho:SHORT-NAME>UInt16</ho:SHORT-NAME><fx:CODING-REF ID-REF="C_UInt16"/></fx:DATATYPE>
      <fx:DATATYPE xsi:type="fx:COMMON-DATATYPE-TYPE" ID="T_Int32"><ho:SHORT-NAME>Int32</ho:SHORT-NAME><fx:CODING-REF ID-REF="C_Int32"/></fx:DATATYPE>
      <fx:DATATYPE xsi:type="fx:COMPLEX-DATATYPE-TYPE" ID="T_Handle">
        <ho:SHORT-NAME>Handle</ho:SHORT-NAME>
        <fx:COMPLEX-DATATYPE-CLASS>TYPEDEF</fx:COMPLEX-DATATYPE-CLASS>
        <fx:MEMBERS><fx:MEMBER ID="T_Handle_v"><ho:SHORT-NAME>v</ho:SHORT-NAME><fx:DATATYPE-REF ID-REF="T_UInt16"/><fx:POSITION>0</fx:POSITION></fx:MEMBER></fx:MEMBERS>
      </fx:DATATYPE>
      <fx:DATATYPE xsi:type="fx:COMPLEX-DATATYPE-TYPE" ID="T_Pair">
        <ho:SHORT-NAME>Pair</ho:SHORT-NAME>
        <fx:COMPLEX-DATATYPE-CLASS>STRUCTURE</fx:COMPLEX-DATATYPE-CLASS>
        <fx:MEMBERS>
          <fx:MEMBER ID="T_Pair_a"><ho:SHORT-NAME>a</ho:SHORT-NAME><fx:DATATYPE-REF ID-REF="T_UInt8"/><fx:POSITION>0</fx:POSITION></fx:MEMBER>
          <fx:MEMBER ID="T_Pair_b"><ho:SHORT-NAME>b</ho:SHORT-NAME><fx:DATATYPE-REF ID-REF="T_UInt16"/><fx:POSITION>1</fx:POSITION></fx:MEMBER>
        </fx:MEMBERS>
      </fx:DATATYPE>
    </fx:DATATYPES>
  </fx:ELEMENTS>
  <fx:PROCESSING-INFORMATION>
    <fx:CODINGS>
      <fx:CODING ID="C_Utf16"><ho:SHORT-NAME>Utf16</ho:SHORT-NAME><ho:CODED-TYPE ho:BASE-DATA-TYPE="A_UNICODE2STRING" CATEGORY="LEADING-LENGTH-INFO-TYPE" ENCODING="UTF-16" TERMINATION="ZERO"/></fx:CODING>
      <fx:CODING ID="C_UInt8"><ho:SHORT-NAME>UInt8</ho:SHORT-NAME><ho:CODED-TYPE ho:BASE-DATA-TYPE="A_UINT8" CATEGORY="STANDARD-LENGTH-TYPE"/></fx:CODING>
      <fx:CODING ID="C_UInt16"><ho:SHORT-NAME>UInt16</ho:SHORT-NAME><ho:CODED-TYPE ho:BASE-DATA-TYPE="A_UINT16" CATEGORY="STANDARD-LENGTH-TYPE"/></fx:CODING>
      <fx:CODING ID="C_Int32"><ho:SHORT-NAME>Int32</ho:SHORT-NAME><ho:CODED-TYPE ho:BASE-DATA-TYPE="A_INT32" CATEGORY="STANDARD-LENGTH-TYPE"/></fx:CODING>
    </fx:CODINGS>
  </fx:PROCESSING-INFORMATION>
</fx:FIBEX>
"#;

    fn load() -> FibexData {
        let dir =
            std::env::temp_dir().join(format!("hilspy-someip-payload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("payload.xml");
        std::fs::write(&path, XML).unwrap();
        let mut fd = FibexData::new();
        fd.load_fibex_file(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        fd
    }

    fn inputs(fd: &FibexData) -> &[Parameter] {
        match &fd.elements.services_map_by_sid_major[&(1, 1)][0].methods_by_mid[&1] {
            MethodIdType::Method(m) => &m.input_params,
            _ => unreachable!(),
        }
    }

    fn encode(ser: &Serializer<'_>, p: &Parameter, v: &Value) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        ser.encode_param(p, Inherited::default(), v, "p", &mut out)?;
        Ok(out)
    }

    fn decode(ser: &Serializer<'_>, p: &Parameter, data: &[u8]) -> Result<Value> {
        ser.decode_param(p, Inherited::default(), &mut Reader::new(data), "p")
    }

    #[test]
    fn layouts() {
        let fd = load();
        let ser = Serializer {
            fd: &fd,
            decode_choices: true,
        };
        let [text, pair, grid, odd] = inputs(&fd) else {
            panic!("four parameters")
        };
        // UTF-16: 16-bit length, BOM, big-endian units, two zero bytes.
        let v = Value::Str("Aé€".into());
        let b = encode(&ser, text, &v).unwrap();
        assert_eq!(b, [0, 10, 0xFE, 0xFF, 0, 0x41, 0, 0xE9, 0x20, 0xAC, 0, 0]);
        assert_eq!(decode(&ser, text, &b).unwrap(), v);
        // A little-endian BOM is honoured.
        let le = [0, 6, 0xFF, 0xFE, 0x41, 0, 0, 0];
        assert_eq!(decode(&ser, text, &le).unwrap(), Value::Str("A".into()));

        // A structure with an 8-bit length field; bytes a newer version
        // appends are skipped.
        let v = Value::Map(vec![
            ("a".into(), Value::Int(1)),
            ("b".into(), Value::Int(0x0203)),
        ]);
        let b = encode(&ser, pair, &v).unwrap();
        assert_eq!(b, [3, 1, 2, 3]);
        let mut r = Reader::new(&[5, 1, 2, 3, 9, 9, 7]);
        assert_eq!(
            ser.decode_param(pair, Inherited::default(), &mut r, "p")
                .unwrap(),
            v
        );
        assert_eq!(r.remaining(), 1);

        // A dynamic array (8-bit length) of fixed pairs of a typedef'd u16.
        let rows = |r: Vec<Vec<u16>>| {
            Value::List(r.into_iter().map(|x| Value::Array(Array::U16(x))).collect())
        };
        let v = rows(vec![vec![1, 2], vec![0xFFFF, 4]]);
        let b = encode(&ser, grid, &v).unwrap();
        assert_eq!(b, [8, 0, 1, 0, 2, 0xFF, 0xFF, 0, 4]);
        assert_eq!(decode(&ser, grid, &b).unwrap(), v);
        let e = encode(&ser, grid, &rows(vec![vec![1]])).unwrap_err();
        assert!(
            e.to_string().contains("p[0]: the array has 2 elements"),
            "{e}"
        );

        // A 24-bit little-endian signed integer, sign extended.
        let b = encode(&ser, odd, &Value::Int(-2)).unwrap();
        assert_eq!(b, [0xFE, 0xFF, 0xFF]);
        assert_eq!(decode(&ser, odd, &b).unwrap(), Value::Int(-2));
        assert!(encode(&ser, odd, &Value::Int(1 << 23)).is_err());
        assert!(decode(&ser, odd, &[1, 2])
            .unwrap_err()
            .to_string()
            .contains("truncated"));
    }

    #[test]
    fn packs_numeric_arrays() {
        assert_eq!(
            pack(Numeric::I64, vec![Value::Int(1), Value::Float(2e19)]),
            Value::List(vec![Value::Int(1), Value::Float(2e19)])
        );
        assert_eq!(
            pack(Numeric::I32, vec![Value::Int(-1)]),
            Value::Array(Array::I32(vec![-1]))
        );
        assert_eq!(int_value(u64::MAX.into()), Value::Float(u64::MAX as f64));
    }
}
