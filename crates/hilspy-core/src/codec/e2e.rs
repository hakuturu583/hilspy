//! AUTOSAR E2E protection of DBC messages (the `e2e` option of `dbc:`),
//! through the `autosar-e2e` crate.
//!
//! ```python
//! hilspy.open("can://can0", "dbc:vehicle.dbc", e2e={
//!     "BrakeCmd": {"profile": 11, "data_id": 0x123, "signals": ["Crc", "Counter"]},
//! })
//! ```
//!
//! Encoding a protected message writes its counter (counting per message)
//! and CRC after the signals are packed; the `signals` the protection writes
//! may be omitted even in strict mode. Decoding checks them and adds an
//! `e2e` field: `Ok`, `CrcError`, `Repeated`, `WrongSequence`,
//! `OkSomeLost`, `DataIdError`, ...
//!
//! Configuration keys (bit offsets and lengths as AUTOSAR defines them;
//! the data length is the message's):
//!
//! | Profile | Keys (defaults) |
//! | --- | --- |
//! | 11 | `data_id` (0x123), `mode` (`nibble` / `both`), `crc_offset` (0), `counter_offset` (8), `nibble_offset` (12) |
//! | 22 | `data_id_list` (16 bytes), `offset` (0) |
//! | 4, 5, 6, 7, 8 | `data_id`, `offset` (0) |
//!
//! Every profile takes `max_delta_counter` (default 1).

use std::collections::BTreeMap;

use autosar_e2e::{
    profile11, profile22, profile4, profile5, profile6, profile7, profile8, E2EProfile, E2EStatus,
};

use crate::error::{Error, Result};
use crate::value::Value;

/// A configured profile: one instance protects, another checks, so a
/// channel can both send and receive the message.
pub(crate) struct Protection {
    sender: Box<dyn Protect>,
    receiver: Box<dyn Protect>,
    /// Signals the protection writes.
    pub(crate) signals: Vec<String>,
}

trait Protect: Send {
    fn protect(&mut self, data: &mut [u8]) -> Result<()>;
    fn check(&mut self, data: &[u8]) -> Result<E2EStatus>;
}

impl<T: E2EProfile + Send> Protect for T {
    fn protect(&mut self, data: &mut [u8]) -> Result<()> {
        E2EProfile::protect(self, data).map_err(|e| Error::encode(format!("E2E: {e}")))
    }

    fn check(&mut self, data: &[u8]) -> Result<E2EStatus> {
        E2EProfile::check(self, data).map_err(|e| Error::decode(format!("E2E: {e}")))
    }
}

pub(crate) fn status_name(s: E2EStatus) -> String {
    format!("{s:?}")
}

/// Typed access to a configuration map; unknown keys are an error.
struct Config {
    message: String,
    keys: BTreeMap<String, Value>,
}

impl Config {
    fn bad(&self, what: impl std::fmt::Display) -> Error {
        Error::Config(format!("e2e config of {}: {what}", self.message))
    }

    fn int<T: TryFrom<i64>>(&mut self, key: &str, default: T) -> Result<T> {
        match self.keys.remove(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v
                .as_i64()
                .and_then(|i| T::try_from(i).ok())
                .ok_or_else(|| self.bad(format!("{key}: {v} is out of range"))),
        }
    }

    fn finish(self) -> Result<()> {
        match self.keys.keys().next() {
            None => Ok(()),
            Some(k) => Err(self.bad(format!("unknown key {k:?}"))),
        }
    }
}

impl Protection {
    /// The protection `config` (a map) describes for a message of
    /// `length` bytes.
    pub(crate) fn new(message: &str, config: Value, length: usize) -> Result<Self> {
        let Value::Map(entries) = config else {
            return Err(Error::Config(format!(
                "e2e config of {message}: expected a mapping, not {config}"
            )));
        };
        let mut c = Config {
            message: message.to_owned(),
            keys: entries.into_iter().collect(),
        };
        let profile: u8 = c.int("profile", 0)?;
        let signals = match c.keys.remove("signals") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::List(l)) => l
                .into_iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s),
                    v => Err(c.bad(format!("signals: {v} is not a signal name"))),
                })
                .collect::<Result<_>>()?,
            Some(v) => return Err(c.bad(format!("signals: {v} is not a list of names"))),
        };
        let bits = length * 8;
        let make = |c: &mut Config| -> Result<Box<dyn Protect>> {
            let err = |c: &Config, e: autosar_e2e::E2EError| c.bad(e);
            Ok(match profile {
                11 => {
                    let mode = match c.keys.remove("mode") {
                        None | Some(Value::Null) => profile11::Profile11IdMode::Nibble,
                        Some(Value::Str(s)) if s.eq_ignore_ascii_case("nibble") => {
                            profile11::Profile11IdMode::Nibble
                        }
                        Some(Value::Str(s)) if s.eq_ignore_ascii_case("both") => {
                            profile11::Profile11IdMode::Both
                        }
                        Some(v) => return Err(c.bad(format!("mode: {v} is not nibble or both"))),
                    };
                    let d = profile11::Profile11Config::default();
                    let config = profile11::Profile11Config {
                        counter_offset: c.int("counter_offset", d.counter_offset)?,
                        crc_offset: c.int("crc_offset", d.crc_offset)?,
                        mode,
                        data_id: c.int("data_id", d.data_id)?,
                        nibble_offset: c.int("nibble_offset", d.nibble_offset)?,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                        data_length: u8::try_from(bits)
                            .map_err(|_| c.bad("profile 11 protects at most 30 bytes"))?,
                    };
                    Box::new(profile11::Profile11::new(config).map_err(|e| err(c, e))?)
                }
                22 => {
                    let list = match c.keys.remove("data_id_list") {
                        Some(v) => v
                            .to_f64_vec()
                            .filter(|l| l.len() == 16)
                            .ok_or_else(|| c.bad("data_id_list: expected 16 numbers"))?,
                        None => return Err(c.bad("profile 22 needs data_id_list (16 numbers)")),
                    };
                    let mut data_id_list = [0u8; 16];
                    for (d, v) in data_id_list.iter_mut().zip(list) {
                        *d = v as u8;
                    }
                    let config = profile22::Profile22Config {
                        data_length: bits,
                        data_id_list,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                        offset: c.int("offset", 0)?,
                    };
                    Box::new(profile22::Profile22::new(config).map_err(|e| err(c, e))?)
                }
                5 => Box::new(
                    profile5::Profile5::new(profile5::Profile5Config {
                        data_length: u16::try_from(bits).map_err(|_| c.bad("too long"))?,
                        data_id: c.int("data_id", 0)?,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                        offset: c.int("offset", 0)?,
                    })
                    .map_err(|e| err(c, e))?,
                ),
                4 => Box::new(
                    profile4::Profile4::new(profile4::Profile4Config {
                        data_id: c.int("data_id", 0)?,
                        offset: c.int("offset", 0)?,
                        min_data_length: u16::try_from(bits).map_err(|_| c.bad("too long"))?,
                        max_data_length: u16::try_from(bits).map_err(|_| c.bad("too long"))?,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                    })
                    .map_err(|e| err(c, e))?,
                ),
                6 => Box::new(
                    profile6::Profile6::new(profile6::Profile6Config {
                        data_id: c.int("data_id", 0)?,
                        offset: c.int("offset", 0)?,
                        min_data_length: u16::try_from(bits).map_err(|_| c.bad("too long"))?,
                        max_data_length: u16::try_from(bits).map_err(|_| c.bad("too long"))?,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                    })
                    .map_err(|e| err(c, e))?,
                ),
                7 => Box::new(
                    profile7::Profile7::new(profile7::Profile7Config {
                        data_id: c.int("data_id", 0)?,
                        offset: c.int("offset", 0)?,
                        min_data_length: bits as u32,
                        max_data_length: bits as u32,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                    })
                    .map_err(|e| err(c, e))?,
                ),
                8 => Box::new(
                    profile8::Profile8::new(profile8::Profile8Config {
                        data_id: c.int("data_id", 0)?,
                        offset: c.int("offset", 0)?,
                        min_data_length: bits as u32,
                        max_data_length: bits as u32,
                        max_delta_counter: c.int("max_delta_counter", 1)?,
                    })
                    .map_err(|e| err(c, e))?,
                ),
                0 => return Err(c.bad("needs a profile (4, 5, 6, 7, 8, 11 or 22)")),
                p => return Err(c.bad(format!("unsupported profile {p} (4, 5, 6, 7, 8, 11, 22)"))),
            })
        };
        // The same keys configure both instances.
        let keys = c.keys.clone();
        let sender = make(&mut c)?;
        c.keys = keys;
        let receiver = make(&mut c)?;
        c.finish()?;
        Ok(Self {
            sender,
            receiver,
            signals,
        })
    }

    pub(crate) fn protect(&mut self, data: &mut [u8]) -> Result<()> {
        self.sender.protect(data)
    }

    pub(crate) fn check(&mut self, data: &[u8]) -> Result<E2EStatus> {
        self.receiver.check(data)
    }
}
