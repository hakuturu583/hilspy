//! Field values of decoded messages and the options of transports and codecs.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};

/// A field value. `Null` is a value the format marks as unavailable (SBF's
/// do-not-use values).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A group of named fields (an optional SBF sub-block, ...).
    Map(Vec<(String, Value)>),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Float(f) if f.fract() == 0.0 && f.is_finite() => Some(*f as i64),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x:?}"),
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Named options of a transport (from the URL query) or a codec (keyword
/// arguments). Readers consume what they know; [`Options::finish`] rejects
/// the rest, so a typo is an error rather than a silent default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    values: BTreeMap<String, Value>,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.values.iter()
    }

    /// Remove and return an option.
    pub fn take(&mut self, key: &str) -> Option<Value> {
        self.values.remove(key)
    }

    pub fn take_bool(&mut self, key: &str, default: bool) -> Result<bool> {
        match self.take(key) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(b),
            Some(Value::Int(i)) => Ok(i != 0),
            Some(Value::Str(s)) => match s.to_ascii_lowercase().as_str() {
                "" | "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(Error::Config(format!(
                    "option {key}: {s:?} is not a boolean"
                ))),
            },
            Some(v) => Err(Error::Config(format!("option {key}: {v} is not a boolean"))),
        }
    }

    pub fn take_str(&mut self, key: &str) -> Result<Option<String>> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::Str(s)) => Ok(Some(s)),
            Some(Value::Int(i)) => Ok(Some(i.to_string())),
            Some(v) => Err(Error::Config(format!("option {key}: {v} is not a string"))),
        }
    }

    /// Fail on options nobody consumed.
    pub fn finish(self, owner: &str) -> Result<()> {
        if self.values.is_empty() {
            return Ok(());
        }
        let names: Vec<_> = self.values.into_keys().collect();
        Err(Error::Config(format!(
            "{owner} does not take the option(s) {}",
            names.join(", ")
        )))
    }
}

impl FromIterator<(String, Value)> for Options {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}
