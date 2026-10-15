//! `hilspy._hilspy`: the PyO3 layer over `hilspy-core`. `hilspy/__init__.py`
//! re-exports everything public.

mod api;
mod convert;
mod errors;
mod plugins;

use pyo3::prelude::*;

#[pymodule]
fn _hilspy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("Error", m.py().get_type::<errors::Error>())?;
    m.add_class::<convert::Message>()?;
    m.add_class::<api::Channel>()?;
    m.add_class::<api::Subscription>()?;
    m.add_class::<api::Periodic>()?;
    m.add_class::<api::Clock>()?;
    m.add_function(wrap_pyfunction!(api::open, m)?)?;
    m.add_function(wrap_pyfunction!(plugins::register_transport, m)?)?;
    m.add_function(wrap_pyfunction!(plugins::register_codec, m)?)?;
    Ok(())
}
