use std::sync::Arc;

use hilspy_core::dbc::{
    ByteOrder, Database as CoreDatabase, DecodeOptions, EncodeOptions, Message as CoreMessage,
    Signal as CoreSignal, Value, ValueKind,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyString};

use crate::can::{data_from, Frame};
use crate::errors::{IntoPyResult, MessageNotFoundError};

fn value_from(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_instance_of::<PyBool>() {
        Ok(Value::Int(i64::from(obj.extract::<bool>()?)))
    } else if obj.is_instance_of::<PyInt>() {
        Ok(Value::Int(obj.extract()?))
    } else if obj.is_instance_of::<PyFloat>() {
        Ok(Value::Float(obj.extract()?))
    } else if obj.is_instance_of::<PyString>() {
        Ok(Value::Str(obj.extract()?))
    } else if let Ok(f) = obj.call_method0("__float__") {
        // numpy scalars and other numbers.
        Ok(Value::Float(f.extract()?))
    } else {
        Err(PyTypeError::new_err(format!(
            "a signal value must be int, float or str, not {}",
            obj.get_type().name()?
        )))
    }
}

fn values_from(values: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Value)>> {
    let mut out = Vec::new();
    let items = if let Ok(d) = values.cast::<PyDict>() {
        d.items().into_any()
    } else {
        values.call_method0("items")?
    };
    for item in items.try_iter()? {
        let (name, value): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.push((name, value_from(&value)?));
    }
    Ok(out)
}

fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
    })
}

fn decoded_to_py<'py>(
    py: Python<'py>,
    decoded: &[(String, Value)],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, v) in decoded {
        d.set_item(name, value_to_py(py, v)?)?;
    }
    Ok(d)
}

/// A DBC database.
#[pyclass(module = "hilspy.dbc", name = "Database", frozen)]
pub(crate) struct Database {
    inner: Arc<CoreDatabase>,
}

/// The message for a name, or an identifier (`extended=None`: 29-bit when
/// above 0x7FF, else the 11-bit message, else the 29-bit one).
fn lookup(db: &CoreDatabase, key: &Bound<'_, PyAny>, extended: Option<bool>) -> PyResult<usize> {
    let message = if let Ok(name) = key.extract::<String>() {
        db.message_by_name(&name).py()?
    } else if let Ok(id) = key.extract::<u32>() {
        match extended {
            Some(e) => db.message_by_id(id, e).py()?,
            None if id > hilspy_core::can::SFF_MASK => db.message_by_id(id, true).py()?,
            None => db
                .message_by_id(id, false)
                .or_else(|_| db.message_by_id(id, true))
                .py()?,
        }
    } else {
        return Err(PyTypeError::new_err(
            "a message is named by a str or an int identifier",
        ));
    };
    Ok(db
        .messages
        .iter()
        .position(|m| std::ptr::eq(m, message))
        .expect("message belongs to the database"))
}

#[pymethods]
impl Database {
    /// Load a DBC file (UTF-8 or Windows-1252).
    #[staticmethod]
    fn from_file(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let db = py.detach(|| CoreDatabase::from_file(path)).py()?;
        Ok(Self {
            inner: Arc::new(db),
        })
    }

    /// Parse DBC text.
    #[staticmethod]
    fn from_string(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(CoreDatabase::parse(text).py()?),
        })
    }

    #[getter]
    fn version(&self) -> String {
        self.inner.version.clone()
    }

    #[getter]
    fn nodes(&self) -> Vec<String> {
        self.inner.nodes.clone()
    }

    #[getter]
    fn messages(&self) -> Vec<Message> {
        (0..self.inner.messages.len())
            .map(|index| Message {
                db: Arc::clone(&self.inner),
                index,
            })
            .collect()
    }

    /// A message by name or identifier.
    #[pyo3(signature = (key, *, extended = None))]
    fn get_message(&self, key: &Bound<'_, PyAny>, extended: Option<bool>) -> PyResult<Message> {
        Ok(Message {
            db: Arc::clone(&self.inner),
            index: lookup(&self.inner, key, extended)?,
        })
    }

    /// Encode signal values (a mapping of name to int, float or value-table
    /// name) into a payload.
    #[pyo3(signature = (key, values, *, scaling = true, strict = true, extended = None))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        scaling: bool,
        strict: bool,
        extended: Option<bool>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.get_message(key, extended)?
            .encode(py, values, scaling, strict)
    }

    /// Encode signal values into a frame with the message's identifier.
    #[pyo3(signature = (key, values, *, scaling = true, strict = true, extended = None))]
    fn encode_frame(
        &self,
        key: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        scaling: bool,
        strict: bool,
        extended: Option<bool>,
    ) -> PyResult<Frame> {
        self.get_message(key, extended)?
            .encode_frame(values, scaling, strict)
    }

    /// Decode a payload of the message named by `key` into a dict.
    #[pyo3(signature = (key, data, *, decode_choices = true, scaling = true, allow_truncated = false, extended = None))]
    #[allow(clippy::too_many_arguments)]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
        data: &Bound<'_, PyAny>,
        decode_choices: bool,
        scaling: bool,
        allow_truncated: bool,
        extended: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.get_message(key, extended)?
            .decode(py, data, decode_choices, scaling, allow_truncated)
    }

    /// Decode a received frame with the message of its identifier.
    #[pyo3(signature = (frame, *, decode_choices = true, scaling = true, allow_truncated = false))]
    fn decode_frame<'py>(
        &self,
        py: Python<'py>,
        frame: &Frame,
        decode_choices: bool,
        scaling: bool,
        allow_truncated: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = DecodeOptions {
            decode_choices,
            scaling,
            allow_truncated,
        };
        let (_, decoded) = self.inner.decode_frame(&frame.inner, &options).py()?;
        decoded_to_py(py, &decoded)
    }

    fn __len__(&self) -> usize {
        self.inner.messages.len()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        match lookup(&self.inner, key, None) {
            Ok(_) => Ok(true),
            Err(e) if e.is_instance_of::<MessageNotFoundError>(key.py()) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Database(version={:?}, messages={})",
            self.inner.version,
            self.inner.messages.len()
        )
    }
}

/// A message of a `Database`.
#[pyclass(module = "hilspy.dbc", name = "Message", frozen)]
pub(crate) struct Message {
    db: Arc<CoreDatabase>,
    index: usize,
}

impl Message {
    fn m(&self) -> &CoreMessage {
        &self.db.messages[self.index]
    }
}

#[pymethods]
impl Message {
    #[getter]
    fn name(&self) -> String {
        self.m().name.clone()
    }

    #[getter]
    fn frame_id(&self) -> u32 {
        self.m().id
    }

    #[getter]
    fn is_extended_frame(&self) -> bool {
        self.m().extended
    }

    /// Payload length in bytes.
    #[getter]
    fn length(&self) -> usize {
        self.m().size
    }

    #[getter]
    fn is_fd(&self) -> bool {
        self.m().fd
    }

    /// `GenMsgCycleTime` in milliseconds.
    #[getter]
    fn cycle_time(&self) -> Option<u64> {
        self.m().cycle_time_ms
    }

    #[getter]
    fn senders(&self) -> Vec<String> {
        self.m().transmitter.iter().cloned().collect()
    }

    #[getter]
    fn comment(&self) -> Option<String> {
        self.m().comment.clone()
    }

    #[getter]
    fn signals(&self) -> Vec<Signal> {
        let m = self.m();
        m.signals.iter().map(|s| Signal::from_core(m, s)).collect()
    }

    fn get_signal(&self, name: &str) -> PyResult<Signal> {
        let m = self.m();
        m.signal(name)
            .map(|s| Signal::from_core(m, s))
            .ok_or_else(|| {
                MessageNotFoundError::new_err(format!("message {} has no signal {name:?}", m.name))
            })
    }

    #[pyo3(signature = (values, *, scaling = true, strict = true))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'_, PyAny>,
        scaling: bool,
        strict: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let values = values_from(values)?;
        let data = self
            .m()
            .encode(&values, &EncodeOptions { scaling, strict })
            .py()?;
        Ok(PyBytes::new(py, &data))
    }

    #[pyo3(signature = (values, *, scaling = true, strict = true))]
    fn encode_frame(
        &self,
        values: &Bound<'_, PyAny>,
        scaling: bool,
        strict: bool,
    ) -> PyResult<Frame> {
        let values = values_from(values)?;
        let inner = self
            .m()
            .encode_frame(&values, &EncodeOptions { scaling, strict })
            .py()?;
        Ok(Frame { inner })
    }

    #[pyo3(signature = (data, *, decode_choices = true, scaling = true, allow_truncated = false))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        decode_choices: bool,
        scaling: bool,
        allow_truncated: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let data = match data.extract::<Frame>() {
            Ok(frame) => frame.inner.data,
            Err(_) => data_from(data)?,
        };
        let options = DecodeOptions {
            decode_choices,
            scaling,
            allow_truncated,
        };
        decoded_to_py(py, &self.m().decode(&data, &options).py()?)
    }

    fn __repr__(&self) -> String {
        let m = self.m();
        format!(
            "Message({:?}, frame_id=0x{:X}, length={}, signals={})",
            m.name,
            m.id,
            m.size,
            m.signals.len()
        )
    }
}

/// A signal definition (a snapshot; immutable).
#[pyclass(module = "hilspy.dbc", name = "Signal", frozen, get_all)]
pub(crate) struct Signal {
    name: String,
    /// Start bit as written in the DBC (LSB for little endian, MSB in
    /// sawtooth numbering for big endian).
    start: u32,
    length: u32,
    /// "little_endian" or "big_endian".
    byte_order: &'static str,
    is_signed: bool,
    is_float: bool,
    scale: f64,
    offset: f64,
    minimum: Option<f64>,
    maximum: Option<f64>,
    unit: String,
    receivers: Vec<String>,
    comment: Option<String>,
    /// Value table: raw value -> name.
    choices: std::collections::BTreeMap<i64, String>,
    /// Raw `GenSigStartValue`.
    initial: Option<f64>,
    is_multiplexer: bool,
    /// Name of the multiplexor this signal depends on.
    multiplexer_signal: Option<String>,
    /// Inclusive multiplexor value ranges for which the signal is present.
    multiplexer_ranges: Vec<(u64, u64)>,
}

impl Signal {
    fn from_core(m: &CoreMessage, s: &CoreSignal) -> Self {
        Self {
            name: s.name.clone(),
            start: s.start_bit,
            length: s.size,
            byte_order: match s.byte_order {
                ByteOrder::LittleEndian => "little_endian",
                ByteOrder::BigEndian => "big_endian",
            },
            is_signed: s.signed,
            is_float: s.kind != ValueKind::Integer,
            scale: s.factor,
            offset: s.offset,
            minimum: s.minimum,
            maximum: s.maximum,
            unit: s.unit.clone(),
            receivers: s.receivers.clone(),
            comment: s.comment.clone(),
            choices: s.choices.iter().cloned().collect(),
            initial: s.initial_raw,
            is_multiplexer: s.is_multiplexor,
            multiplexer_signal: s
                .multiplexing
                .as_ref()
                .map(|x| m.signals[x.multiplexor].name.clone()),
            multiplexer_ranges: s
                .multiplexing
                .as_ref()
                .map(|x| x.ranges.clone())
                .unwrap_or_default(),
        }
    }
}

#[pymethods]
impl Signal {
    fn __repr__(&self) -> String {
        format!(
            "Signal({:?}, start={}, length={}, byte_order={:?}, scale={}, offset={}, unit={:?})",
            self.name, self.start, self.length, self.byte_order, self.scale, self.offset, self.unit
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_class::<Message>()?;
    m.add_class::<Signal>()?;
    Ok(())
}
