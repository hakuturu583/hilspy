use std::sync::Mutex;
use std::time::{Duration, Instant};

use hilspy_core::can::{
    Bus as CoreBus, BusConfig, Filter as CoreFilter, Frame as CoreFrame,
    PeriodicTask as CorePeriodic,
};
use pyo3::exceptions::{PyStopIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::errors::{BusClosedError, IntoPyResult};

/// How long a blocking receive runs without the GIL before checking for
/// KeyboardInterrupt.
const SIGNAL_CHECK: Duration = Duration::from_millis(50);

/// A CAN or CAN FD frame (immutable).
#[pyclass(module = "hilspy.can", name = "Frame", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub(crate) struct Frame {
    pub(crate) inner: CoreFrame,
}

pub(crate) fn data_from(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyTypeError::new_err(
            "data must be bytes-like or a sequence of ints, not str",
        ));
    }
    obj.extract::<Vec<u8>>().map_err(|_| {
        PyTypeError::new_err("data must be bytes-like or a sequence of ints in 0..=255")
    })
}

#[pymethods]
impl Frame {
    /// Create a frame. `is_extended_id=None` picks a 29-bit identifier when
    /// `arbitration_id` exceeds 0x7FF.
    #[new]
    #[pyo3(signature = (arbitration_id, data = None, *, is_extended_id = None, is_fd = false, bitrate_switch = false, is_remote_frame = false, dlc = None))]
    fn new(
        arbitration_id: u32,
        data: Option<&Bound<'_, PyAny>>,
        is_extended_id: Option<bool>,
        is_fd: bool,
        bitrate_switch: bool,
        is_remote_frame: bool,
        dlc: Option<u8>,
    ) -> PyResult<Self> {
        let extended = is_extended_id.unwrap_or(arbitration_id > hilspy_core::can::SFF_MASK);
        let data = data.map(data_from).transpose()?.unwrap_or_default();
        let inner = if is_remote_frame {
            if is_fd || !data.is_empty() {
                return Err(PyValueError::new_err(
                    "a remote frame has no data and cannot be CAN FD",
                ));
            }
            CoreFrame::new_remote(arbitration_id, dlc.unwrap_or(0), extended)
        } else {
            if dlc.is_some_and(|d| usize::from(d) != data.len()) {
                return Err(PyValueError::new_err("dlc differs from len(data)"));
            }
            if is_fd {
                CoreFrame::new_fd(arbitration_id, &data, extended, bitrate_switch)
            } else {
                if bitrate_switch {
                    return Err(PyValueError::new_err("bitrate_switch needs is_fd=True"));
                }
                CoreFrame::new(arbitration_id, &data, extended)
            }
        }
        .py()?;
        Ok(Self { inner })
    }

    #[getter]
    fn arbitration_id(&self) -> u32 {
        self.inner.id
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }

    #[getter]
    fn dlc(&self) -> u8 {
        self.inner.dlc
    }

    #[getter]
    fn is_extended_id(&self) -> bool {
        self.inner.extended
    }

    #[getter]
    fn is_remote_frame(&self) -> bool {
        self.inner.remote
    }

    #[getter]
    fn is_error_frame(&self) -> bool {
        self.inner.error
    }

    #[getter]
    fn is_fd(&self) -> bool {
        self.inner.fd
    }

    #[getter]
    fn bitrate_switch(&self) -> bool {
        self.inner.brs
    }

    #[getter]
    fn error_state_indicator(&self) -> bool {
        self.inner.esi
    }

    /// Reception time in seconds since the Unix epoch (None if not received).
    #[getter]
    fn timestamp(&self) -> Option<f64> {
        self.inner.timestamp
    }

    fn __len__(&self) -> usize {
        self.inner.data.len()
    }

    fn __repr__(&self) -> String {
        let f = &self.inner;
        let id = if f.extended {
            format!("0x{:08X}", f.id)
        } else {
            format!("0x{:03X}", f.id)
        };
        let mut flags = String::new();
        if f.fd {
            flags.push_str(", is_fd=True");
            if f.brs {
                flags.push_str(", bitrate_switch=True");
            }
        }
        if f.extended && f.id <= hilspy_core::can::SFF_MASK {
            flags.push_str(", is_extended_id=True");
        }
        if f.remote {
            return format!("Frame({id}, is_remote_frame=True, dlc={}{flags})", f.dlc);
        }
        if f.error {
            return format!(
                "Frame({id}, is_error_frame=True, data=bytes.fromhex('{}'))",
                hex(&f.data)
            );
        }
        format!("Frame({id}, bytes.fromhex('{}'){flags})", hex(&f.data))
    }
}

fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

/// A receive filter: a frame passes when `id & can_mask == can_id & can_mask`.
#[pyclass(module = "hilspy.can", name = "Filter", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub(crate) struct Filter {
    inner: CoreFilter,
}

#[pymethods]
impl Filter {
    /// `extended`: True for 29-bit identifiers only, False for 11-bit only,
    /// None for both. `can_mask` defaults to an exact match.
    #[new]
    #[pyo3(signature = (can_id, can_mask = None, extended = None))]
    fn new(can_id: u32, can_mask: Option<u32>, extended: Option<bool>) -> Self {
        let mask = can_mask.unwrap_or(if extended == Some(false) {
            hilspy_core::can::SFF_MASK
        } else {
            hilspy_core::can::EFF_MASK
        });
        Self {
            inner: CoreFilter {
                id: can_id,
                mask,
                extended,
            },
        }
    }

    #[getter]
    fn can_id(&self) -> u32 {
        self.inner.id
    }

    #[getter]
    fn can_mask(&self) -> u32 {
        self.inner.mask
    }

    #[getter]
    fn extended(&self) -> Option<bool> {
        self.inner.extended
    }

    fn matches(&self, frame: &Frame) -> bool {
        self.inner.matches(&frame.inner)
    }

    fn __repr__(&self) -> String {
        let ext = match self.inner.extended {
            None => "None",
            Some(true) => "True",
            Some(false) => "False",
        };
        format!(
            "Filter(can_id=0x{:X}, can_mask=0x{:X}, extended={ext})",
            self.inner.id, self.inner.mask
        )
    }
}

/// Filters from `Filter` objects or python-can style dicts
/// (`{"can_id": ..., "can_mask": ..., "extended": ...}`).
fn filters_from(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<CoreFilter>> {
    let Some(obj) = obj else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        if let Ok(f) = item.extract::<Filter>() {
            out.push(f.inner);
        } else if let Ok(d) = item.cast::<PyDict>() {
            let get = |k: &str| d.get_item(k);
            let can_id: u32 = get("can_id")?
                .ok_or_else(|| PyValueError::new_err("a filter dict needs can_id"))?
                .extract()?;
            let can_mask: Option<u32> = get("can_mask")?.map(|v| v.extract()).transpose()?;
            let extended: Option<bool> = match get("extended")? {
                Some(v) if !v.is_none() => Some(v.extract()?),
                _ => None,
            };
            out.push(Filter::new(can_id, can_mask, extended).inner);
        } else {
            return Err(PyTypeError::new_err(
                "filters must be Filter objects or dicts",
            ));
        }
    }
    Ok(out)
}

fn duration(seconds: f64, what: &str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err(format!("invalid {what}: {seconds}")))
}

/// A CAN bus. `interface="socketcan"` opens a Linux SocketCAN interface
/// (`can0`, `vcan0`); `interface="virtual"` joins an in-process bus shared by
/// every `Bus` opened with the same channel name.
#[pyclass(module = "hilspy.can", name = "Bus", frozen)]
pub(crate) struct Bus {
    inner: CoreBus,
}

#[pymethods]
impl Bus {
    #[new]
    #[pyo3(signature = (channel, interface = "socketcan", *, fd = false, receive_own_messages = false, filters = None))]
    fn new(
        py: Python<'_>,
        channel: String,
        interface: &str,
        fd: bool,
        receive_own_messages: bool,
        filters: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut config = BusConfig::new(interface, channel);
        config.fd = fd;
        config.receive_own_messages = receive_own_messages;
        config.filters = filters_from(filters)?;
        let inner = py.detach(|| CoreBus::open(config)).py()?;
        Ok(Self { inner })
    }

    #[getter]
    fn channel(&self) -> String {
        self.inner.config().channel.clone()
    }

    #[getter]
    fn interface(&self) -> String {
        self.inner.config().interface.clone()
    }

    #[getter]
    fn fd(&self) -> bool {
        self.inner.config().fd
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn send(&self, py: Python<'_>, frame: &Frame) -> PyResult<()> {
        let f = frame.inner.clone();
        py.detach(|| self.inner.send(&f)).py()
    }

    /// Wait for a frame; None once `timeout` seconds pass (`None` waits
    /// forever, Ctrl-C interrupts).
    #[pyo3(signature = (timeout = None))]
    fn recv(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Option<Frame>> {
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
            if let Some(f) = py.detach(|| self.inner.recv(Some(slice))).py()? {
                return Ok(Some(Frame { inner: f }));
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
            py.check_signals()?;
        }
    }

    fn set_filters(&self, filters: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.set_filters(&filters_from(filters)?).py()
    }

    /// Send `frame` every `period` seconds from a background thread (for
    /// `duration` seconds, or until stopped).
    #[pyo3(signature = (frame, period, duration = None))]
    fn send_periodic(
        &self,
        frame: &Frame,
        period: f64,
        duration: Option<f64>,
    ) -> PyResult<PeriodicTask> {
        let period = self::duration(period, "period")?;
        let total = duration
            .map(|d| self::duration(d, "duration"))
            .transpose()?;
        let task = self
            .inner
            .send_periodic(frame.inner.clone(), period, total)
            .py()?;
        Ok(PeriodicTask {
            inner: Mutex::new(task),
        })
    }

    /// Close the bus; blocked `recv` calls raise BusClosedError.
    fn shutdown(&self) {
        self.inner.shutdown();
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _exc_type: Py<PyAny>, _exc: Py<PyAny>, _tb: Py<PyAny>) -> bool {
        self.inner.shutdown();
        false
    }

    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Iterate over received frames until the bus is shut down.
    fn __next__(&self, py: Python<'_>) -> PyResult<Frame> {
        match self.recv(py, None) {
            Ok(Some(f)) => Ok(f),
            Ok(None) => unreachable!("recv without a timeout returns a frame"),
            Err(e) if e.is_instance_of::<BusClosedError>(py) => Err(PyStopIteration::new_err(())),
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "Bus(channel={:?}, interface={:?}, fd={}{})",
            c.channel,
            c.interface,
            if c.fd { "True" } else { "False" },
            if self.inner.is_closed() {
                ", closed"
            } else {
                ""
            }
        )
    }
}

/// A frame sent periodically by `Bus.send_periodic`.
#[pyclass(module = "hilspy.can", name = "PeriodicTask", frozen)]
pub(crate) struct PeriodicTask {
    inner: Mutex<CorePeriodic>,
}

impl PeriodicTask {
    fn task(&self) -> std::sync::MutexGuard<'_, CorePeriodic> {
        self.inner.lock().unwrap()
    }

    fn running(&self) -> PyResult<std::sync::MutexGuard<'_, CorePeriodic>> {
        let task = self.task();
        if task.is_running() {
            Ok(task)
        } else {
            Err(BusClosedError::new_err("the periodic task has stopped"))
        }
    }
}

#[pymethods]
impl PeriodicTask {
    /// Stop sending and wait for the sender thread (idempotent).
    fn stop(&self, py: Python<'_>) {
        let mut guard = self.task();
        let task: &mut CorePeriodic = &mut guard;
        py.detach(|| task.stop());
    }

    /// Send `frame` instead from the next period on.
    fn modify(&self, frame: &Frame) -> PyResult<()> {
        self.running()?.modify(frame.inner.clone()).py()
    }

    #[getter]
    fn frame(&self) -> Frame {
        Frame {
            inner: self.task().frame(),
        }
    }

    /// Period in seconds.
    #[getter]
    fn get_period(&self) -> f64 {
        self.task().period().as_secs_f64()
    }

    #[setter]
    fn set_period(&self, period: f64) -> PyResult<()> {
        let period = duration(period, "period")?;
        self.running()?.set_period(period).py()
    }

    #[getter]
    fn sent_count(&self) -> u64 {
        self.task().sent_count()
    }

    /// The last send error message (the task keeps running after one).
    #[getter]
    fn last_error(&self) -> Option<String> {
        self.task().last_error()
    }

    #[getter]
    fn is_running(&self) -> bool {
        self.task().is_running()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Py<PyAny>,
        _exc: Py<PyAny>,
        _tb: Py<PyAny>,
    ) -> bool {
        self.stop(py);
        false
    }

    fn __repr__(&self) -> String {
        let t = self.task();
        format!(
            "PeriodicTask(period={}, sent_count={}, running={})",
            t.period().as_secs_f64(),
            t.sent_count(),
            if t.is_running() { "True" } else { "False" }
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Frame>()?;
    m.add_class::<Filter>()?;
    m.add_class::<Bus>()?;
    m.add_class::<PeriodicTask>()?;
    m.add("SFF_MASK", hilspy_core::can::SFF_MASK)?;
    m.add("EFF_MASK", hilspy_core::can::EFF_MASK)?;
    Ok(())
}
