//! `open`, `Channel` and `Periodic`.

use std::time::{Duration, Instant};

use hilspy_core::{Channel as CoreChannel, Error as CoreError, Periodic as CorePeriodic};
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{message_arg, options_from_py, Message};
use crate::errors::{to_py, IntoPyResult};

/// How long a blocking receive runs without the GIL before checking for
/// KeyboardInterrupt.
const SIGNAL_CHECK: Duration = Duration::from_millis(50);

fn duration(seconds: f64, what: &str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err(format!("invalid {what}: {seconds}")))
}

/// Open a channel: the transport `url` with the codec `codec`.
///
/// Transports: `can://vcan0` (Linux SocketCAN; `?fd`, `?echo`,
/// `?filter=0x100:0x700`), `udp://host:port` (`?listen`, `?bind=`),
/// `tcp://host:port` (`?listen`), `mem://name` (in-process). Codecs: `raw`,
/// `dbc:FILE` (options `strict`, `scaling`, `decode_choices`), `sbf`.
/// Keyword arguments configure the codec.
#[pyfunction]
#[pyo3(signature = (url, codec = String::from("raw"), **options))]
pub(crate) fn open(
    py: Python<'_>,
    url: String,
    codec: String,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Channel> {
    let options = options_from_py(options)?;
    let inner = py
        .detach(|| CoreChannel::open(&url, &codec, options))
        .py()?;
    Ok(Channel { inner, url, codec })
}

/// A transport and a codec composed: send and receive messages.
#[pyclass(module = "hilspy", name = "Channel", frozen)]
pub(crate) struct Channel {
    inner: CoreChannel,
    url: String,
    codec: String,
}

#[pymethods]
impl Channel {
    /// Send a message: a `Message`, or a name and its fields.
    #[pyo3(signature = (message, fields = None))]
    fn send(
        &self,
        py: Python<'_>,
        message: &Bound<'_, PyAny>,
        fields: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let message = message_arg(message, fields)?;
        py.detach(|| self.inner.send(&message)).py()
    }

    /// The next message; None once `timeout` seconds pass (None waits
    /// forever; Ctrl-C interrupts).
    #[pyo3(signature = (timeout = None))]
    fn recv(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Option<Message>> {
        let deadline = timeout
            .map(|t| duration(t.max(0.0), "timeout"))
            .transpose()?
            .map(|t| Instant::now() + t);
        loop {
            let slice = match deadline {
                None => SIGNAL_CHECK,
                Some(d) => d
                    .saturating_duration_since(Instant::now())
                    .min(SIGNAL_CHECK),
            };
            if let Some(m) = py.detach(|| self.inner.recv(Some(slice))).py()? {
                return Ok(Some(Message::from_core(m)));
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
            py.check_signals()?;
        }
    }

    /// Send a message every `period` seconds from a background thread, for
    /// `duration` seconds or until stopped.
    #[pyo3(signature = (period, message, fields = None, *, duration = None))]
    fn every(
        &self,
        py: Python<'_>,
        period: f64,
        message: &Bound<'_, PyAny>,
        fields: Option<&Bound<'_, PyAny>>,
        duration: Option<f64>,
    ) -> PyResult<Periodic> {
        let message = message_arg(message, fields)?;
        let period = self::duration(period, "period")?;
        let total = duration
            .map(|d| self::duration(d, "duration"))
            .transpose()?;
        // Without the GIL: encoding locks the codec, which a periodic thread
        // running a Python codec may hold while it waits for the GIL.
        let task = py
            .detach(|| self.inner.every(period, message, total))
            .py()?;
        Ok(Periodic { inner: task })
    }

    /// Close the channel; blocked receivers stop, iteration ends.
    fn close(&self) {
        self.inner.close();
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _t: Py<PyAny>, _e: Py<PyAny>, _tb: Py<PyAny>) -> bool {
        self.inner.close();
        false
    }

    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Received messages until the channel closes.
    fn __next__(&self, py: Python<'_>) -> PyResult<Message> {
        loop {
            match py.detach(|| self.inner.recv(Some(SIGNAL_CHECK))) {
                Ok(Some(m)) => return Ok(Message::from_core(m)),
                Ok(None) => py.check_signals()?,
                Err(CoreError::Closed) => return Err(PyStopIteration::new_err(())),
                Err(e) => return Err(to_py(e)),
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Channel({:?}, {:?}{})",
            self.url,
            self.codec,
            if self.inner.is_closed() {
                ", closed"
            } else {
                ""
            }
        )
    }
}

/// A message sent periodically by `Channel.every`.
///
/// No method holds a lock across a GIL-dependent wait: the sender thread may
/// need the GIL (a Python codec or transport) while `stop()` joins it.
#[pyclass(module = "hilspy", name = "Periodic", frozen)]
pub(crate) struct Periodic {
    inner: CorePeriodic,
}

impl Drop for Periodic {
    /// Stop without the GIL: the sender thread may be waiting for it inside
    /// a Python codec or transport.
    fn drop(&mut self) {
        let task = &self.inner;
        Python::attach(|py| py.detach(|| task.stop()));
    }
}

#[pymethods]
impl Periodic {
    /// Send this message instead from the next period on.
    #[pyo3(signature = (message, fields = None))]
    fn update(
        &self,
        py: Python<'_>,
        message: &Bound<'_, PyAny>,
        fields: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let message = message_arg(message, fields)?;
        py.detach(|| self.inner.update(message));
        Ok(())
    }

    /// Stop sending and wait for the sender thread (idempotent).
    fn stop(&self, py: Python<'_>) {
        py.detach(|| self.inner.stop());
    }

    /// Period in seconds.
    #[getter]
    fn get_period(&self) -> f64 {
        self.inner.period().as_secs_f64()
    }

    #[setter]
    fn set_period(&self, period: f64) -> PyResult<()> {
        self.inner.set_period(duration(period, "period")?).py()
    }

    #[getter]
    fn sent_count(&self) -> u64 {
        self.inner.sent_count()
    }

    /// The last send error (sending continues after one).
    #[getter]
    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.is_running()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, py: Python<'_>, _t: Py<PyAny>, _e: Py<PyAny>, _tb: Py<PyAny>) -> bool {
        self.stop(py);
        false
    }

    fn __repr__(&self) -> String {
        let t = &self.inner;
        format!(
            "Periodic(period={}, sent_count={}, running={})",
            t.period().as_secs_f64(),
            t.sent_count(),
            if t.is_running() { "True" } else { "False" }
        )
    }
}
