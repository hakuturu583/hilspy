//! `hilspy._hilspy`: the PyO3 layer over `hilspy-core`. The Python package
//! re-exports these classes from `hilspy.can`, `hilspy.dbc` and `hilspy.sbf`.

mod can;
mod dbc;
mod errors;
mod sbf;

use pyo3::prelude::*;

#[pymodule]
fn _hilspy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(m)?;
    can::register(m)?;
    dbc::register(m)?;
    sbf::register(m)?;
    Ok(())
}
