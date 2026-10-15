//! Field values of decoded messages and the options of transports and codecs.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};

/// A field value. `Null` is a value the format marks as unavailable (SBF's
/// do-not-use values).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A group of named fields (an optional SBF sub-block, ...).
    Map(Vec<(String, Value)>),
    /// A homogeneous numeric array (a point cloud column, ...).
    Array(Array),
}

/// A typed numeric array; Python sees an `array.array`.
#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(),
            Array::U16(v) => v.len(),
            Array::U32(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::F32(v) => v.len(),
            Array::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as `f64` (exact except for `I64` beyond 2^53).
    pub fn to_f64(&self) -> Vec<f64> {
        fn conv<T: Copy>(v: &[T], f: impl Fn(T) -> f64) -> Vec<f64> {
            v.iter().map(|x| f(*x)).collect()
        }
        match self {
            Array::U8(v) => conv(v, f64::from),
            Array::U16(v) => conv(v, f64::from),
            Array::U32(v) => conv(v, f64::from),
            Array::I32(v) => conv(v, f64::from),
            Array::I64(v) => conv(v, |x| x as f64),
            Array::F32(v) => conv(v, f64::from),
            Array::F64(v) => v.clone(),
        }
    }

    /// The `array` module type code of the element type.
    pub fn typecode(&self) -> char {
        match self {
            Array::U8(_) => 'B',
            Array::U16(_) => 'H',
            Array::U32(_) => 'I',
            Array::I32(_) => 'i',
            Array::I64(_) => 'q',
            Array::F32(_) => 'f',
            Array::F64(_) => 'd',
        }
    }
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    /// The numbers of an array, or of a single number.
    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            Value::Array(a) => Some(a.to_f64()),
            v => v.as_f64().map(|x| vec![x]),
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Float(f) if f.fract() == 0.0 && f.is_finite() => Some(*f as i64),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x:?}"),
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
            Value::Array(a) => write!(f, "array({:?}, <{} items>)", a.typecode(), a.len()),
        }
    }
}

/// Named options of a transport (from the URL query) or a codec (keyword
/// arguments). Readers consume what they know; [`Options::finish`] rejects
/// the rest, so a typo is an error rather than a silent default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    values: BTreeMap<String, Value>,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.values.iter()
    }

    /// Remove and return an option.
    pub fn take(&mut self, key: &str) -> Option<Value> {
        self.values.remove(key)
    }

    pub fn take_bool(&mut self, key: &str, default: bool) -> Result<bool> {
        match self.take(key) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(b),
            Some(Value::Int(i)) => Ok(i != 0),
            Some(Value::Str(s)) => match s.to_ascii_lowercase().as_str() {
                "" | "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(Error::Config(format!(
                    "option {key}: {s:?} is not a boolean"
                ))),
            },
            Some(v) => Err(Error::Config(format!("option {key}: {v} is not a boolean"))),
        }
    }

    pub fn take_str(&mut self, key: &str) -> Result<Option<String>> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::Str(s)) => Ok(Some(s)),
            Some(Value::Int(i)) => Ok(Some(i.to_string())),
            Some(v) => Err(Error::Config(format!("option {key}: {v} is not a string"))),
        }
    }

    pub fn take_f64(&mut self, key: &str) -> Result<Option<f64>> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::Str(s)) => s
                .trim()
                .parse()
                .map(Some)
                .map_err(|_| Error::Config(format!("option {key}: {s:?} is not a number"))),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| Error::Config(format!("option {key}: {v} is not a number"))),
        }
    }

    /// Fail on options nobody consumed.
    pub fn finish(self, owner: &str) -> Result<()> {
        if self.values.is_empty() {
            return Ok(());
        }
        let names: Vec<_> = self.values.into_keys().collect();
        Err(Error::Config(format!(
            "{owner} does not take the option(s) {}",
            names.join(", ")
        )))
    }
}

impl FromIterator<(String, Value)> for Options {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}
