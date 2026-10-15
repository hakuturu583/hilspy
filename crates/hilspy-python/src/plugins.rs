//! Transports and codecs written in Python, registered into the core
//! registries so `open()` composes them like the built-in ones.

use std::sync::Arc;
use std::time::Duration;

use hilspy_core::codec::{self, Codec};
use hilspy_core::transport::{self, Transport};
use hilspy_core::{Message as CoreMessage, Packet, Result as CoreResult};
use pyo3::prelude::*;

use crate::convert::{options_to_py, Message};
use crate::errors::external;

fn packet_to_py(py: Python<'_>, packet: &Packet) -> PyResult<Py<Message>> {
    Py::new(py, Message::from_core(CoreMessage::raw(packet.clone())))
}

fn message_from(obj: &Bound<'_, PyAny>) -> PyResult<CoreMessage> {
    Ok(obj.extract::<Message>()?.inner)
}

/// A list of `Message`s (also accepts one Message or None).
fn messages_from(obj: &Bound<'_, PyAny>) -> PyResult<Vec<CoreMessage>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    if let Ok(m) = obj.extract::<Message>() {
        return Ok(vec![m.inner]);
    }
    obj.try_iter()?.map(|m| message_from(&m?)).collect()
}

struct PyTransport {
    obj: Py<PyAny>,
}

impl Transport for PyTransport {
    fn send(&self, packet: &Packet) -> CoreResult<()> {
        Python::attach(|py| {
            let m = packet_to_py(py, packet)?;
            self.obj.call_method1(py, "send", (m,))?;
            Ok(())
        })
        .map_err(external)
    }

    fn recv(&self, timeout: Option<Duration>) -> CoreResult<Option<Packet>> {
        Python::attach(|py| -> PyResult<Option<Packet>> {
            let r = self
                .obj
                .call_method1(py, "recv", (timeout.map(|t| t.as_secs_f64()),))?;
            let r = r.bind(py);
            if r.is_none() {
                return Ok(None);
            }
            Ok(Some(r.extract::<Message>()?.packet()?))
        })
        .map_err(external)
    }

    fn close(&self) {
        Python::attach(|py| {
            if self.obj.bind(py).hasattr("close").unwrap_or(false) {
                if let Err(e) = self.obj.call_method0(py, "close") {
                    e.write_unraisable(py, Some(self.obj.bind(py)));
                }
            }
        });
    }
}

struct PyCodec {
    obj: Py<PyAny>,
}

impl Codec for PyCodec {
    fn encode(&mut self, message: &CoreMessage) -> CoreResult<Vec<Packet>> {
        Python::attach(|py| -> PyResult<Vec<Packet>> {
            let m = Py::new(py, Message::from_core(message.clone()))?;
            let r = self.obj.call_method1(py, "encode", (m,))?;
            messages_from(r.bind(py))?
                .into_iter()
                .map(|m| Message::from_core(m).packet())
                .collect()
        })
        .map_err(external)
    }

    fn decode(&mut self, packet: Packet) -> CoreResult<Vec<CoreMessage>> {
        Python::attach(|py| -> PyResult<Vec<CoreMessage>> {
            let m = packet_to_py(py, &packet)?;
            let r = self.obj.call_method1(py, "decode", (m,))?;
            let mut out = messages_from(r.bind(py))?;
            // Decoded messages keep the packet they came from by default.
            for m in &mut out {
                if m.packet.is_none() {
                    m.packet = Some(packet.clone());
                }
            }
            Ok(out)
        })
        .map_err(external)
    }
}

/// Register a transport for a URL scheme. `factory(address, options)`
/// gets the part of the URL after `scheme://` (without the query) and the
/// query as a dict, and returns an object with `send(message)`,
/// `recv(timeout) -> Message | None` and optionally `close()`. Packets are
/// nameless `Message`s (`data`, and `id` for CAN).
#[pyfunction]
pub(crate) fn register_transport(scheme: String, factory: Py<PyAny>) {
    transport::register(
        &scheme,
        Arc::new(move |endpoint| {
            Python::attach(|py| -> PyResult<Arc<dyn Transport>> {
                let options = options_to_py(py, &endpoint.options)?;
                let obj = factory.call1(py, (endpoint.address, options))?;
                Ok(Arc::new(PyTransport { obj }))
            })
            .map_err(external)
        }),
    );
}

/// Register a codec. `factory(argument, options)` gets the text after
/// `name:` in the codec spec (or None) and the keyword arguments of
/// `open()`, and returns an object with `encode(message) -> list[Message]`
/// (nameless messages carrying the packets) and
/// `decode(packet) -> list[Message]`.
#[pyfunction]
pub(crate) fn register_codec(name: String, factory: Py<PyAny>) {
    codec::register(
        &name,
        Arc::new(move |spec| {
            Python::attach(|py| -> PyResult<Box<dyn Codec>> {
                let options = options_to_py(py, &spec.options)?;
                let obj = factory.call1(py, (spec.argument, options))?;
                Ok(Box::new(PyCodec { obj }))
            })
            .map_err(external)
        }),
    );
}
