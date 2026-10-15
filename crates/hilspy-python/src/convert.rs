//! `Message` and conversions between Python objects and core values.

use hilspy_core::{
    Array, CanHeader, FlexRayChannel, FlexRayHeader, LinChecksum, LinHeader,
    Message as CoreMessage, Options, Packet, Value,
};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

const SFF_MASK: u32 = 0x7FF;

pub(crate) fn value_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if obj.is_instance_of::<PyBool>() {
        Ok(Value::Bool(obj.extract()?))
    } else if obj.is_instance_of::<PyInt>() {
        match obj.extract::<i64>() {
            Ok(i) => Ok(Value::Int(i)),
            // An unsigned 64-bit value above i64::MAX.
            Err(e) => obj.extract::<u64>().map(Value::UInt).map_err(|_| e),
        }
    } else if obj.is_instance_of::<PyFloat>() {
        Ok(Value::Float(obj.extract()?))
    } else if obj.is_instance_of::<PyString>() {
        Ok(Value::Str(obj.extract()?))
    } else if obj.hasattr("items")? {
        Ok(Value::Map(fields_from_py(obj)?))
    } else if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        sequence_from_py(obj)
    } else if let Some(a) = array_from_py(obj)? {
        Ok(Value::Array(a))
    } else if let Ok(f) = obj.call_method0("__float__") {
        // numpy scalars and other numbers.
        Ok(Value::Float(f.extract()?))
    } else {
        Err(PyTypeError::new_err(format!(
            "a field value must be None, bool, int, float, str, a mapping or a sequence \
             of numbers (list, array.array, numpy array), not {}",
            obj.get_type().name()?
        )))
    }
}

/// A list or tuple: an int64 array if all are ints, a float64 array if all
/// are numbers, else a list of values.
fn sequence_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    if items
        .iter()
        .all(|i| i.is_instance_of::<PyInt>() && !i.is_instance_of::<PyBool>())
    {
        let v = items.iter().map(|i| i.extract()).collect::<PyResult<_>>()?;
        return Ok(Value::Array(Array::I64(v)));
    }
    let numbers = items
        .iter()
        .all(|i| !i.is_instance_of::<PyString>() && i.extract::<f64>().is_ok());
    if numbers {
        let v = items.iter().map(|i| i.extract()).collect::<PyResult<_>>()?;
        return Ok(Value::Array(Array::F64(v)));
    }
    // Anything else (dicts, strings, nested lists) is a list of values.
    let v = items.iter().map(value_from_py).collect::<PyResult<_>>()?;
    Ok(Value::List(v))
}

fn chunks<const N: usize, T>(b: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
    b.chunks_exact(N)
        .map(|c| f(c.try_into().expect("exact chunk")))
        .collect()
}

/// An `array.array` or a numpy array (anything with `typecode` or a native
/// `dtype`, `itemsize` and `tobytes()`), as a typed array.
fn array_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Option<Array>> {
    let (code, size) = if obj.hasattr("typecode")? && obj.hasattr("tobytes")? {
        (
            obj.getattr("typecode")?.extract::<String>()?,
            obj.getattr("itemsize")?.extract::<usize>()?,
        )
    } else if obj.hasattr("dtype")? && obj.hasattr("tobytes")? {
        if obj.getattr("ndim")?.extract::<usize>()? == 0 {
            return Ok(None); // a numpy scalar
        }
        let dtype = obj.getattr("dtype")?;
        if !dtype.getattr("isnative")?.extract::<bool>()? {
            return Err(PyValueError::new_err(
                "arrays must be in native byte order (use arr.astype(arr.dtype.newbyteorder('=')))",
            ));
        }
        (
            dtype.getattr("char")?.extract::<String>()?,
            dtype.getattr("itemsize")?.extract::<usize>()?,
        )
    } else {
        return Ok(None);
    };
    let bytes = obj.call_method0("tobytes")?;
    let b = bytes.cast::<PyBytes>()?.as_bytes();
    let signed = "bhilq".contains(code.as_str());
    let unsigned = "?BHILQ".contains(code.as_str());
    Ok(Some(match (code.as_str(), size) {
        ("f", 4) => Array::F32(chunks(b, f32::from_ne_bytes)),
        ("d", 8) => Array::F64(chunks(b, f64::from_ne_bytes)),
        (_, 1) if unsigned => Array::U8(b.to_vec()),
        (_, 1) if signed => Array::I32(b.iter().map(|x| i32::from(*x as i8)).collect()),
        (_, 2) if unsigned => Array::U16(chunks(b, u16::from_ne_bytes)),
        (_, 2) if signed => Array::I32(chunks(b, |x| i32::from(i16::from_ne_bytes(x)))),
        (_, 4) if unsigned => Array::U32(chunks(b, u32::from_ne_bytes)),
        (_, 4) if signed => Array::I32(chunks(b, i32::from_ne_bytes)),
        (_, 8) if signed => Array::I64(chunks(b, i64::from_ne_bytes)),
        (_, 8) if unsigned => Array::F64(chunks(b, |x| u64::from_ne_bytes(x) as f64)),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unsupported array element type {code:?} ({size} bytes)"
            )))
        }
    }))
}

fn array_to_py<'py>(py: Python<'py>, a: &Array) -> PyResult<Bound<'py, PyAny>> {
    fn ne<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
        v.iter().flat_map(|x| f(*x)).collect()
    }
    let bytes = match a {
        Array::U8(v) => v.clone(),
        Array::U16(v) => ne(v, u16::to_ne_bytes),
        Array::U32(v) => ne(v, u32::to_ne_bytes),
        Array::I32(v) => ne(v, i32::to_ne_bytes),
        Array::I64(v) => ne(v, i64::to_ne_bytes),
        Array::F32(v) => ne(v, f32::to_ne_bytes),
        Array::F64(v) => ne(v, f64::to_ne_bytes),
    };
    let out = py
        .import("array")?
        .getattr("array")?
        .call1((a.typecode().to_string(),))?;
    out.call_method1("frombytes", (PyBytes::new(py, &bytes),))?;
    Ok(out)
}

pub(crate) fn fields_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Value)>> {
    let items = match obj.cast::<PyDict>() {
        Ok(d) => d.items().into_any(),
        Err(_) => obj.call_method0("items")?,
    };
    let mut out = Vec::new();
    for item in items.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.push((k, value_from_py(&v)?));
    }
    Ok(out)
}

pub(crate) fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::UInt(u) => u.into_pyobject(py)?.into_any(),
        Value::Float(x) => x.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Map(m) => fields_to_py(py, m)?.into_any(),
        Value::Array(a) => array_to_py(py, a)?,
        Value::List(l) => {
            let items = l
                .iter()
                .map(|v| value_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
    })
}

pub(crate) fn fields_to_py<'py>(
    py: Python<'py>,
    fields: &[(String, Value)],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in fields {
        d.set_item(k, value_to_py(py, v)?)?;
    }
    Ok(d)
}

pub(crate) fn options_from_py(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Options> {
    let mut o = Options::new();
    if let Some(kwargs) = kwargs {
        for (k, v) in kwargs.iter() {
            o.insert(k.extract::<String>()?, value_from_py(&v)?);
        }
    }
    Ok(o)
}

pub(crate) fn options_to_py<'py>(
    py: Python<'py>,
    options: &Options,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in options.iter() {
        d.set_item(k, value_to_py(py, v)?)?;
    }
    Ok(d)
}

fn bytes_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("data must be bytes-like, not str"));
    }
    obj.extract::<Vec<u8>>().map_err(|_| {
        PyTypeError::new_err("data must be bytes-like or a sequence of ints in 0..=255")
    })
}

/// A message: a name and fields (decoded or to encode), and the raw packet
/// (`data`, plus `id` and the bus header for CAN, LIN and FlexRay frames) it
/// was received as or is sent as.
///
/// `Message("VehicleSpeed", {"Speed": 12.5})` is encoded by the channel's
/// codec; `Message(data=b"...", id=0x123)` (no name) is sent as is.
#[pyclass(module = "hilspy", name = "Message", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub(crate) struct Message {
    pub(crate) inner: CoreMessage,
}

impl Message {
    pub(crate) fn from_core(inner: CoreMessage) -> Self {
        Self { inner }
    }

    pub(crate) fn packet(&self) -> PyResult<Packet> {
        self.inner
            .packet
            .clone()
            .ok_or_else(|| PyValueError::new_err("the message has no data"))
    }
}

/// A `Message`, or a name with optional fields, as a core message.
pub(crate) fn message_arg(
    message: &Bound<'_, PyAny>,
    fields: Option<&Bound<'_, PyAny>>,
) -> PyResult<CoreMessage> {
    if let Ok(m) = message.extract::<Message>() {
        if fields.is_some() {
            return Err(PyTypeError::new_err(
                "fields go with a message name, not a Message",
            ));
        }
        return Ok(m.inner);
    }
    let name: String = message
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a Message or a message name"))?;
    let fields = fields.map(fields_from_py).transpose()?.unwrap_or_default();
    Ok(CoreMessage::new(name, fields))
}

#[pymethods]
impl Message {
    #[new]
    #[pyo3(signature = (
        name = None, fields = None, *, data = None, id = None, bus = None, extended = None,
        fd = false, enhanced = None, checksum = None, cycle = None, channel = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: Option<String>,
        fields: Option<&Bound<'_, PyAny>>,
        data: Option<&Bound<'_, PyAny>>,
        id: Option<u32>,
        bus: Option<&str>,
        extended: Option<bool>,
        fd: bool,
        enhanced: Option<bool>,
        checksum: Option<u8>,
        cycle: Option<u8>,
        channel: Option<&str>,
    ) -> PyResult<Self> {
        let fields = fields.map(fields_from_py).transpose()?.unwrap_or_default();
        let data = data.map(bytes_from_py).transpose()?;
        let lin_args = enhanced.is_some() || checksum.is_some();
        let flexray_args = cycle.is_some() || channel.is_some();
        let bus = match bus {
            Some(b) => Some(match b.to_ascii_lowercase().as_str() {
                "can" => "can",
                "lin" => "lin",
                "flexray" => "flexray",
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "bus must be \"can\", \"lin\" or \"flexray\", not {b:?}"
                    )))
                }
            }),
            None if lin_args => Some("lin"),
            None if flexray_args => Some("flexray"),
            None => id.map(|_| "can"),
        };
        let misplaced = match bus {
            Some("can") => {
                (lin_args || flexray_args).then_some("enhanced, checksum, cycle and channel")
            }
            Some("lin") => (extended.is_some() || fd || flexray_args)
                .then_some("extended, fd, cycle and channel"),
            Some(_) => (extended.is_some() || fd || lin_args)
                .then_some("extended, fd, enhanced and checksum"),
            None => (extended.is_some() || fd).then_some("extended and fd"),
        };
        if let Some(what) = misplaced {
            return Err(PyValueError::new_err(match bus {
                Some(b) => format!("{what} do not apply to a {b} frame"),
                None => format!("{what} need an id"),
            }));
        }
        if bus.is_some() && id.is_none() {
            return Err(PyValueError::new_err(
                "a CAN, LIN or FlexRay frame needs an id",
            ));
        }
        let mut packet = if data.is_some() || id.is_some() {
            Some(Packet {
                data: data.unwrap_or_default(),
                ..Packet::default()
            })
        } else {
            None
        };
        if let (Some(p), Some(id), Some(bus)) = (packet.as_mut(), id, bus) {
            match bus {
                "can" => {
                    p.can = Some(CanHeader {
                        id,
                        extended: extended.unwrap_or(id > SFF_MASK),
                        fd,
                        brs: fd,
                    })
                }
                "lin" => {
                    let id = u8::try_from(id)
                        .ok()
                        .filter(|id| *id <= 0x3F)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!("LIN identifier {id} is not in 0..=63"))
                        })?;
                    // Diagnostic frames always use the classic checksum.
                    let model = match enhanced {
                        _ if hilspy_core::is_lin_diagnostic(id) => LinChecksum::Classic,
                        Some(true) => LinChecksum::Enhanced,
                        Some(false) => LinChecksum::Classic,
                        None => LinChecksum::for_id(id),
                    };
                    p.lin = Some(LinHeader {
                        id,
                        checksum_model: model,
                        checksum: checksum.unwrap_or_else(|| model.compute(id, &p.data)),
                    });
                }
                _ => {
                    let slot = u16::try_from(id).map_err(|_| {
                        PyValueError::new_err(format!("FlexRay slot {id} is not in 1..=2047"))
                    })?;
                    let channel = match channel {
                        None => FlexRayChannel::A,
                        Some(c) => FlexRayChannel::parse(c).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "channel must be \"A\", \"B\" or \"AB\", not {c:?}"
                            ))
                        })?,
                    };
                    p.flexray = Some(FlexRayHeader {
                        slot,
                        cycle: cycle.unwrap_or(0),
                        channel,
                    });
                }
            }
            p.check_headers()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if name.is_none() && packet.is_none() {
            return Err(PyValueError::new_err("a message needs a name or data"));
        }
        Ok(Self {
            inner: CoreMessage {
                name,
                fields,
                packet,
            },
        })
    }

    /// The message name; None for a raw (undecoded) message.
    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    /// Decoded fields (a new dict on each access).
    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        fields_to_py(py, &self.inner.fields)
    }

    /// The raw bytes (a CAN payload, an SBF block, a datagram), if any.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .packet
            .as_ref()
            .map(|p| PyBytes::new(py, &p.data))
    }

    /// The bus a frame is from: "can", "lin" or "flexray"; None for byte
    /// packets.
    #[getter]
    fn bus(&self) -> Option<&'static str> {
        let p = self.inner.packet.as_ref()?;
        if p.can.is_some() {
            Some("can")
        } else if p.lin.is_some() {
            Some("lin")
        } else if p.flexray.is_some() {
            Some("flexray")
        } else {
            None
        }
    }

    /// The frame identifier: CAN identifier, LIN frame identifier (0..=63)
    /// or FlexRay slot.
    #[getter]
    fn id(&self) -> Option<u32> {
        let p = self.inner.packet.as_ref()?;
        p.can
            .map(|h| h.id)
            .or(p.lin.map(|h| u32::from(h.id)))
            .or(p.flexray.map(|h| u32::from(h.slot)))
    }

    /// LIN frame sent with the enhanced checksum model (else classic).
    #[getter]
    fn enhanced(&self) -> bool {
        self.lin()
            .is_some_and(|h| h.checksum_model == LinChecksum::Enhanced)
    }

    /// The checksum byte of a LIN frame.
    #[getter]
    fn checksum(&self) -> Option<u8> {
        self.lin().map(|h| h.checksum)
    }

    /// The cycle counter (0..=63) of a FlexRay frame.
    #[getter]
    fn cycle(&self) -> Option<u8> {
        self.flexray().map(|h| h.cycle)
    }

    /// The channel of a FlexRay frame: "A", "B" or "AB".
    #[getter]
    fn channel(&self) -> Option<&'static str> {
        self.flexray().map(|h| h.channel.name())
    }

    /// 29-bit CAN identifier.
    #[getter]
    fn extended(&self) -> bool {
        self.can().is_some_and(|h| h.extended)
    }

    /// CAN FD frame.
    #[getter]
    fn fd(&self) -> bool {
        self.can().is_some_and(|h| h.fd)
    }

    /// Reception time, seconds since the Unix epoch.
    #[getter]
    fn timestamp(&self) -> Option<f64> {
        self.inner.timestamp()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.field(key) {
            Some(v) => value_to_py(py, v),
            None => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.field(key).is_some()
    }

    fn __repr__(&self) -> String {
        let m = &self.inner;
        let mut parts = Vec::new();
        if let Some(n) = &m.name {
            parts.push(format!("{n:?}"));
        }
        if !m.fields.is_empty() {
            let f: Vec<_> = m
                .fields
                .iter()
                .map(|(k, v)| format!("{k:?}: {v}"))
                .collect();
            parts.push(format!("{{{}}}", f.join(", ")));
        }
        if let Some(p) = &m.packet {
            if let Some(h) = p.can {
                parts.push(format!("id=0x{:X}", h.id));
            }
            if let Some(h) = p.lin {
                parts.push(format!("bus=\"lin\", id=0x{:02X}", h.id));
            }
            if let Some(h) = p.flexray {
                parts.push(format!(
                    "bus=\"flexray\", id={}, cycle={}, channel={:?}",
                    h.slot,
                    h.cycle,
                    h.channel.name()
                ));
            }
            if m.name.is_none() {
                let hex: String = p.data.iter().map(|b| format!("{b:02x}")).collect();
                parts.push(format!("data=bytes.fromhex('{hex}')"));
            }
        }
        format!("Message({})", parts.join(", "))
    }
}

impl Message {
    fn can(&self) -> Option<CanHeader> {
        self.inner.packet.as_ref()?.can
    }

    fn lin(&self) -> Option<LinHeader> {
        self.inner.packet.as_ref()?.lin
    }

    fn flexray(&self) -> Option<FlexRayHeader> {
        self.inner.packet.as_ref()?.flexray
    }
}
