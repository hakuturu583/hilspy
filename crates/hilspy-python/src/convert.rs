//! `Message` and conversions between Python objects and core values.

use hilspy_core::{CanHeader, Message as CoreMessage, Options, Packet, Value};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyString};

const SFF_MASK: u32 = 0x7FF;

pub(crate) fn value_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if obj.is_instance_of::<PyBool>() {
        Ok(Value::Bool(obj.extract()?))
    } else if obj.is_instance_of::<PyInt>() {
        Ok(Value::Int(obj.extract()?))
    } else if obj.is_instance_of::<PyFloat>() {
        Ok(Value::Float(obj.extract()?))
    } else if obj.is_instance_of::<PyString>() {
        Ok(Value::Str(obj.extract()?))
    } else if obj.hasattr("items")? {
        Ok(Value::Map(fields_from_py(obj)?))
    } else if let Ok(f) = obj.call_method0("__float__") {
        // numpy scalars and other numbers.
        Ok(Value::Float(f.extract()?))
    } else {
        Err(PyTypeError::new_err(format!(
            "a field value must be None, bool, int, float, str or a mapping, not {}",
            obj.get_type().name()?
        )))
    }
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
        Value::Float(x) => x.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Map(m) => fields_to_py(py, m)?.into_any(),
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
/// (`data`, plus `id` for CAN) it was received as or is sent as.
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
    #[pyo3(signature = (name = None, fields = None, *, data = None, id = None, extended = None, fd = false))]
    fn new(
        name: Option<String>,
        fields: Option<&Bound<'_, PyAny>>,
        data: Option<&Bound<'_, PyAny>>,
        id: Option<u32>,
        extended: Option<bool>,
        fd: bool,
    ) -> PyResult<Self> {
        let fields = fields.map(fields_from_py).transpose()?.unwrap_or_default();
        let data = data.map(bytes_from_py).transpose()?;
        if id.is_none() && (extended.is_some() || fd) {
            return Err(PyValueError::new_err("extended and fd need an id"));
        }
        let packet = if data.is_some() || id.is_some() {
            Some(Packet {
                data: data.unwrap_or_default(),
                can: id.map(|id| CanHeader {
                    id,
                    extended: extended.unwrap_or(id > SFF_MASK),
                    fd,
                    brs: fd,
                }),
                timestamp: None,
            })
        } else {
            None
        };
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

    /// CAN identifier, for CAN frames.
    #[getter]
    fn id(&self) -> Option<u32> {
        self.inner.packet.as_ref()?.can.map(|h| h.id)
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
}
