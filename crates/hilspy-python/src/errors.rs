use hilspy_core::Error as CoreError;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    hilspy,
    Error,
    PyException,
    "A hilspy error: a bad URL, codec or option, a message that cannot be \
     encoded or decoded, or a closed channel. Operating-system errors are \
     raised as OSError; errors from Python plugins propagate unchanged."
);

pub(crate) fn to_py(e: CoreError) -> PyErr {
    match e {
        // Keeps the specific subclass (FileNotFoundError, ConnectionRefusedError, ...).
        CoreError::Io(e) => PyErr::from(e),
        CoreError::External(e) => match e.downcast::<PyErr>() {
            Ok(py_err) => *py_err,
            Err(other) => Error::new_err(other.to_string()),
        },
        other => Error::new_err(other.to_string()),
    }
}

pub(crate) trait IntoPyResult<T> {
    fn py(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for hilspy_core::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(to_py)
    }
}

/// A Python exception as a core error, re-raised unchanged on the way back.
pub(crate) fn external(e: PyErr) -> CoreError {
    CoreError::External(Box::new(e))
}
