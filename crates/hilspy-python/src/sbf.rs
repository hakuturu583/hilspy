use hilspy_core::sbf::{self, encode, Block as CoreBlock, Decoder as CoreDecoder, Field};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::can::data_from;
use crate::errors::IntoPyResult;

fn field_to_py<'py>(py: Python<'py>, f: &Field) -> PyResult<Bound<'py, PyAny>> {
    Ok(match f {
        Field::Null => py.None().into_bound(py),
        Field::Bool(b) => pyo3::types::PyBool::new(py, *b).to_owned().into_any(),
        Field::Int(i) => i.into_pyobject(py)?.into_any(),
        Field::Float(x) => x.into_pyobject(py)?.into_any(),
        Field::Str(s) => s.into_pyobject(py)?.into_any(),
        Field::Map(m) => fields_to_py(py, m)?.into_any(),
    })
}

fn fields_to_py<'py>(py: Python<'py>, fields: &[(String, Field)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in fields {
        d.set_item(k, field_to_py(py, v)?)?;
    }
    Ok(d)
}

/// One SBF block.
#[pyclass(module = "hilspy.sbf", name = "Block", frozen)]
pub(crate) struct Block {
    inner: CoreBlock,
}

#[pymethods]
impl Block {
    /// Block number (4007 = PVTGeodetic, ...).
    #[getter]
    fn number(&self) -> u16 {
        self.inner.number
    }

    #[getter]
    fn revision(&self) -> u8 {
        self.inner.revision
    }

    /// Block name, or None for a block the parser does not know.
    #[getter]
    fn name(&self) -> Option<&'static str> {
        self.inner.name
    }

    /// GPS time of week in milliseconds (None: do-not-use).
    #[getter]
    fn tow(&self) -> Option<u32> {
        self.inner.tow
    }

    /// GPS week number (None: do-not-use).
    #[getter]
    fn wnc(&self) -> Option<u16> {
        self.inner.wnc
    }

    /// The whole block, sync to padding.
    #[getter]
    fn raw<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.raw)
    }

    /// The block body (after the 8-byte header).
    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.body())
    }

    /// Parsed fields (do-not-use values are None), or None for blocks
    /// without a field mapping.
    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .fields
            .as_ref()
            .map(|f| fields_to_py(py, f))
            .transpose()
    }

    /// Why a known block's body could not be parsed.
    #[getter]
    fn parse_error(&self) -> Option<String> {
        self.inner.parse_error.clone()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.field(key) {
            Some(f) => field_to_py(py, f),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        let b = &self.inner;
        let opt = |v: Option<String>| v.unwrap_or_else(|| "None".into());
        format!(
            "Block({}, number={}, revision={}, tow={}, wnc={})",
            b.name
                .map_or_else(|| "None".to_owned(), |n| format!("{n:?}")),
            b.number,
            b.revision,
            opt(b.tow.map(|t| t.to_string())),
            opt(b.wnc.map(|w| w.to_string())),
        )
    }
}

/// A streaming SBF decoder: feed bytes in any chunks, get whole blocks.
#[pyclass(module = "hilspy.sbf", name = "Decoder")]
pub(crate) struct Decoder {
    inner: CoreDecoder,
}

#[pymethods]
impl Decoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreDecoder::new(),
        }
    }

    /// Append bytes; return the blocks they complete.
    fn feed(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Vec<Block>> {
        let data = data_from(data)?;
        let inner = &mut self.inner;
        let blocks = py.detach(|| inner.feed(&data));
        Ok(blocks.into_iter().map(|inner| Block { inner }).collect())
    }

    /// Blocks decoded, CRC errors, and bytes skipped between blocks.
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("blocks", s.blocks)?;
        d.set_item("crc_errors", s.crc_errors)?;
        d.set_item("skipped_bytes", s.skipped_bytes)?;
        Ok(d)
    }

    /// Bytes held waiting for the rest of a block.
    #[getter]
    fn pending(&self) -> usize {
        self.inner.pending()
    }
}

/// Decode every block in a complete buffer.
#[pyfunction]
fn decode(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Vec<Block>> {
    let data = data_from(data)?;
    let blocks = py.detach(|| sbf::decode_all(&data));
    Ok(blocks.into_iter().map(|inner| Block { inner }).collect())
}

/// CRC-16/XMODEM as SBF uses.
#[pyfunction]
fn crc16(data: &Bound<'_, PyAny>) -> PyResult<u16> {
    Ok(sbf::crc16(&data_from(data)?))
}

/// Add the SBF header to a block body (which starts with TOW and WNc) and
/// pad it to a multiple of 4 bytes.
#[pyfunction]
fn frame_block<'py>(
    py: Python<'py>,
    number: u16,
    revision: u8,
    body: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(
        py,
        &sbf::frame_block(number, revision, &data_from(body)?).py()?,
    ))
}

/// Encode a PVTGeodetic block (4007 rev 2). Latitude/longitude in radians,
/// height in metres, velocities in m/s, accuracies in cm; None = do-not-use.
#[pyfunction]
#[pyo3(signature = (*, tow = None, wnc = None, mode = 0, error = 0, latitude = None, longitude = None, height = None, undulation = None, vn = None, ve = None, vu = None, cog = None, rx_clk_bias = None, rx_clk_drift = None, time_system = Some(0), datum = Some(0), nr_sv = None, wa_corr_info = 0, reference_id = None, mean_corr_age = None, signal_info = 0, alert_flag = 0, nr_bases = 0, ppp_info = 0, latency = None, h_accuracy = None, v_accuracy = None, misc = 0))]
#[allow(clippy::too_many_arguments)]
fn encode_pvt_geodetic<'py>(
    py: Python<'py>,
    tow: Option<u32>,
    wnc: Option<u16>,
    mode: u8,
    error: u8,
    latitude: Option<f64>,
    longitude: Option<f64>,
    height: Option<f64>,
    undulation: Option<f64>,
    vn: Option<f64>,
    ve: Option<f64>,
    vu: Option<f64>,
    cog: Option<f64>,
    rx_clk_bias: Option<f64>,
    rx_clk_drift: Option<f64>,
    time_system: Option<u8>,
    datum: Option<u8>,
    nr_sv: Option<u8>,
    wa_corr_info: u8,
    reference_id: Option<u16>,
    mean_corr_age: Option<u16>,
    signal_info: u32,
    alert_flag: u8,
    nr_bases: u8,
    ppp_info: u16,
    latency: Option<u16>,
    h_accuracy: Option<u16>,
    v_accuracy: Option<u16>,
    misc: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let block = encode::PvtGeodetic {
        tow,
        wnc,
        mode,
        error,
        latitude,
        longitude,
        height,
        undulation,
        vn,
        ve,
        vu,
        cog,
        rx_clk_bias,
        rx_clk_drift,
        time_system,
        datum,
        nr_sv,
        wa_corr_info,
        reference_id,
        mean_corr_age,
        signal_info,
        alert_flag,
        nr_bases,
        ppp_info,
        latency,
        h_accuracy,
        v_accuracy,
        misc,
    };
    Ok(PyBytes::new(py, &block.encode().py()?))
}

/// Encode an AttEuler block (5938 rev 1). Angles in degrees, rates in deg/s;
/// None = do-not-use.
#[pyfunction]
#[pyo3(signature = (*, tow = None, wnc = None, nr_sv = None, error = 0, mode = 0, heading = None, pitch = None, roll = None, pitch_dot = None, roll_dot = None, heading_dot = None))]
#[allow(clippy::too_many_arguments)]
fn encode_att_euler<'py>(
    py: Python<'py>,
    tow: Option<u32>,
    wnc: Option<u16>,
    nr_sv: Option<u8>,
    error: u8,
    mode: u16,
    heading: Option<f64>,
    pitch: Option<f64>,
    roll: Option<f64>,
    pitch_dot: Option<f64>,
    roll_dot: Option<f64>,
    heading_dot: Option<f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let block = encode::AttEuler {
        tow,
        wnc,
        nr_sv,
        error,
        mode,
        heading,
        pitch,
        roll,
        pitch_dot,
        roll_dot,
        heading_dot,
    };
    Ok(PyBytes::new(py, &block.encode().py()?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Block>()?;
    m.add_class::<Decoder>()?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(crc16, m)?)?;
    m.add_function(wrap_pyfunction!(frame_block, m)?)?;
    m.add_function(wrap_pyfunction!(encode_pvt_geodetic, m)?)?;
    m.add_function(wrap_pyfunction!(encode_att_euler, m)?)?;
    Ok(())
}
