//! `open`, `Channel` and `Periodic`.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use hilspy_core::{
    Channel as CoreChannel, Clock as CoreClock, Error as CoreError, LinkDirection,
    Periodic as CorePeriodic, Subscription as CoreSubscription,
};
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::convert::{message_arg, options_from_py, Message};
use crate::errors::{to_py, IntoPyResult};

/// How long a blocking receive runs without the GIL before checking for
/// KeyboardInterrupt.
const SIGNAL_CHECK: Duration = Duration::from_millis(50);

fn duration(seconds: f64, what: &str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err(format!("invalid {what}: {seconds}")))
}

/// A direction argument: `"send"`, `"recv"` or `"both"`.
fn link(direction: &str) -> PyResult<LinkDirection> {
    direction.parse().py()
}

/// Open a channel: the transport `url` with the codec `codec`.
///
/// Transports: `can://vcan0` (Linux SocketCAN; `?fd`, `?echo`,
/// `?filter=0x100:0x700`), `udp://host:port` (`?listen`, `?bind=`),
/// `tcp://host:port` (`?listen`), `mem://name` (in-process). Codecs: `raw`,
/// `dbc:FILE` (options `strict`, `scaling`, `decode_choices`, `e2e`, `ecu`),
/// `fibex:FILE` (option `channel`, and the `dbc` options), `sbf`, ...
/// Keyword arguments configure the codec.
#[pyfunction(name = "_open")]
#[pyo3(signature = (url, codec = String::from("raw"), *, clock = None, **options))]
pub(crate) fn open(
    py: Python<'_>,
    url: String,
    codec: String,
    clock: Option<PyRef<'_, Clock>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Channel> {
    let options = options_from_py(options)?;
    let clock = clock.map(|c| c.inner.clone());
    let inner = py
        .detach(|| CoreChannel::open_on(&url, &codec, options, clock))
        .py()?;
    Ok(Channel { inner, url, codec })
}

/// Export the metrics recorded so far and stop exporting (`atexit`; see
/// `hilspy_core::telemetry`). A no-op unless `OTEL_EXPORTER_OTLP_*` turned
/// exporting on.
#[pyfunction(name = "_shutdown_telemetry")]
pub(crate) fn shutdown_telemetry(py: Python<'_>) {
    py.detach(hilspy_core::telemetry::shutdown);
}

/// A transport and a codec composed: send and receive messages.
#[pyclass(module = "hilspy._hilspy", name = "_Channel", frozen)]
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
    /// `duration` seconds or until stopped. `period=None` uses the message's
    /// cycle time from the codec's network description (DBC, FIBEX).
    #[pyo3(signature = (period, message, fields = None, *, duration = None))]
    fn every(
        &self,
        py: Python<'_>,
        period: Option<f64>,
        message: &Bound<'_, PyAny>,
        fields: Option<&Bound<'_, PyAny>>,
        duration: Option<f64>,
    ) -> PyResult<Periodic> {
        let message = message_arg(message, fields)?;
        let period = period.map(|p| self::duration(p, "period")).transpose()?;
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

    /// Receive on a background thread and call `callback(item)` for every
    /// item (with the GIL): a `Message`, an exception instance for a receive
    /// error, or None once when the channel closes. A falsy return value or
    /// an exception from the callback ends the subscription.
    fn subscribe(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<Subscription> {
        let deliver = move |item: hilspy_core::Result<hilspy_core::Message>| -> bool {
            Python::attach(|py| {
                let arg = match item {
                    Ok(m) => match Py::new(py, Message::from_core(m)) {
                        Ok(m) => m.into_any(),
                        Err(e) => e.into_value(py).into_any(),
                    },
                    Err(CoreError::Closed) => py.None(),
                    Err(e) => to_py(e).into_value(py).into_any(),
                };
                match callback.call1(py, (arg,)) {
                    Ok(r) => r.bind(py).is_truthy().unwrap_or(false),
                    Err(e) => {
                        e.write_unraisable(py, Some(callback.bind(py)));
                        false
                    }
                }
            })
        };
        let inner = py.detach(|| self.inner.subscribe(deliver)).py()?;
        Ok(Subscription { inner })
    }

    /// Simulate losing the link: what travels in `direction` (`"send"`,
    /// `"recv"` or `"both"`) is lost until `reconnect()`, or for `duration`
    /// seconds (on the channel's clock if it has one); with `messages`, only
    /// the messages of those names. The transport stays open; sends and
    /// periodic messages still encode, then drop their packets.
    #[pyo3(signature = (duration = None, *, direction = "both", messages = None))]
    fn disconnect(
        &self,
        duration: Option<f64>,
        direction: &str,
        messages: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let direction = link(direction)?;
        let duration = duration
            .map(|d| self::duration(d, "duration"))
            .transpose()?;
        // One name, or an iterable of names.
        let messages = match messages {
            None => None,
            Some(m) if m.is_instance_of::<PyString>() => Some(BTreeSet::from([m.extract()?])),
            Some(m) => Some(
                m.try_iter()?
                    .map(|n| n?.extract::<String>())
                    .collect::<PyResult<_>>()?,
            ),
        };
        self.inner.disconnect(direction, duration, messages).py()
    }

    /// Bring `direction` of the link back up.
    #[pyo3(signature = (*, direction = "both"))]
    fn reconnect(&self, direction: &str) -> PyResult<()> {
        let direction = link(direction)?;
        self.inner.reconnect(direction).py()
    }

    /// Whether `direction` of the link is down (either side, for `"both"`).
    #[pyo3(signature = (direction = "both"))]
    fn disconnected(&self, direction: &str) -> PyResult<bool> {
        let direction = link(direction)?;
        Ok(self.inner.is_disconnected(direction))
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

/// A running `_Channel.subscribe`.
#[pyclass(module = "hilspy._hilspy", name = "_Subscription", frozen)]
pub(crate) struct Subscription {
    inner: CoreSubscription,
}

impl Drop for Subscription {
    /// Stop without the GIL: the receive thread may be waiting for it to
    /// deliver a message.
    fn drop(&mut self) {
        let sub = &self.inner;
        Python::attach(|py| py.detach(|| sub.stop()));
    }
}

#[pymethods]
impl Subscription {
    /// Stop delivering and wait for the receive thread (idempotent).
    fn stop(&self, py: Python<'_>) {
        py.detach(|| self.inner.stop());
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.is_running()
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

/// Simulated time for synchronous simulation.
///
/// Channels opened with `clock=` run on it: `every()` sends and the paced
/// packets of a message happen inside `advance()`, in time order, and
/// messages are stamped with its time.
#[pyclass(module = "hilspy", name = "Clock", frozen)]
pub(crate) struct Clock {
    pub(crate) inner: CoreClock,
}

#[pymethods]
impl Clock {
    #[new]
    #[pyo3(signature = (start = 0.0))]
    fn new(start: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CoreClock::new(start).py()?,
        })
    }

    /// The current time, seconds.
    #[getter]
    fn now(&self) -> f64 {
        self.inner.now()
    }

    /// When the next scheduled send is due, or None.
    #[getter]
    fn next_due(&self) -> Option<f64> {
        self.inner.next_due()
    }

    /// Advance by `dt` seconds, sending everything due on the way.
    fn advance(&self, py: Python<'_>, dt: f64) -> PyResult<()> {
        py.detach(|| self.inner.advance(dt)).py()
    }

    /// Advance to `t` seconds, sending everything due on the way.
    fn advance_to(&self, py: Python<'_>, t: f64) -> PyResult<()> {
        py.detach(|| self.inner.advance_to(t)).py()
    }

    fn __repr__(&self) -> String {
        format!(
            "Clock(now={}, pending={})",
            self.inner.now(),
            self.inner.pending()
        )
    }
}
