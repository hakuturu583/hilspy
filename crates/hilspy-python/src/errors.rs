use hilspy_core::Error;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    hilspy,
    HilspyError,
    PyException,
    "Base class of hilspy errors."
);
create_exception!(
    hilspy.can,
    InvalidFrameError,
    HilspyError,
    "A frame that cannot exist on the bus."
);
create_exception!(
    hilspy.can,
    BusClosedError,
    HilspyError,
    "The bus was shut down."
);
create_exception!(
    hilspy.can,
    UnsupportedInterfaceError,
    HilspyError,
    "An unknown or unavailable bus interface."
);
create_exception!(
    hilspy.dbc,
    DbcError,
    HilspyError,
    "A DBC file that does not parse or is inconsistent."
);
create_exception!(
    hilspy.dbc,
    MessageNotFoundError,
    DbcError,
    "A message or signal not in the database."
);
create_exception!(
    hilspy.dbc,
    EncodeError,
    DbcError,
    "Signal values that cannot be encoded."
);
create_exception!(
    hilspy.dbc,
    DecodeError,
    DbcError,
    "A payload that cannot be decoded."
);
create_exception!(hilspy.sbf, SbfError, HilspyError, "Invalid SBF input.");

pub(crate) fn to_py(e: Error) -> PyErr {
    match e {
        // Keeps the specific subclass (FileNotFoundError, PermissionError, ...).
        Error::Io(e) => PyErr::from(e),
        Error::InvalidFrame(_) => InvalidFrameError::new_err(e.to_string()),
        Error::BusClosed => BusClosedError::new_err(e.to_string()),
        Error::Unsupported(_) => UnsupportedInterfaceError::new_err(e.to_string()),
        Error::DbcParse(_) => DbcError::new_err(e.to_string()),
        Error::NotFound(_) => MessageNotFoundError::new_err(e.to_string()),
        Error::Encode(_) => EncodeError::new_err(e.to_string()),
        Error::Decode(_) => DecodeError::new_err(e.to_string()),
        Error::Sbf(_) => SbfError::new_err(e.to_string()),
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("HilspyError", py.get_type::<HilspyError>())?;
    m.add("InvalidFrameError", py.get_type::<InvalidFrameError>())?;
    m.add("BusClosedError", py.get_type::<BusClosedError>())?;
    m.add(
        "UnsupportedInterfaceError",
        py.get_type::<UnsupportedInterfaceError>(),
    )?;
    m.add("DbcError", py.get_type::<DbcError>())?;
    m.add(
        "MessageNotFoundError",
        py.get_type::<MessageNotFoundError>(),
    )?;
    m.add("EncodeError", py.get_type::<EncodeError>())?;
    m.add("DecodeError", py.get_type::<DecodeError>())?;
    m.add("SbfError", py.get_type::<SbfError>())?;
    Ok(())
}
