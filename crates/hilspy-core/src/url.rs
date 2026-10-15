//! `scheme://address?key=value&flag` endpoints and `name:argument` codec
//! specifications.

use crate::error::{Error, Result};
use crate::value::{Options, Value};

/// A parsed transport URL.
#[derive(Clone, Debug, PartialEq)]
pub struct Endpoint {
    pub scheme: String,
    /// Everything between `://` and `?` (an interface, `host:port`, a name).
    pub address: String,
    /// The query, values as strings (a bare `flag` is an empty string,
    /// which reads as true).
    pub options: Options,
}

impl Endpoint {
    pub fn parse(url: &str) -> Result<Self> {
        let (scheme, rest) = url.split_once("://").ok_or_else(|| {
            Error::Config(format!(
                "{url:?} is not a transport URL (scheme://address, e.g. can://vcan0, \
                 udp://192.168.0.10:5000, tcp://host:28784, mem://bus)"
            ))
        })?;
        if scheme.is_empty()
            || !scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
        {
            return Err(Error::Config(format!("invalid URL scheme {scheme:?}")));
        }
        let (address, query) = rest.split_once('?').unwrap_or((rest, ""));
        let mut options = Options::new();
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            options.insert(k, Value::Str(v.to_owned()));
        }
        Ok(Self {
            scheme: scheme.to_ascii_lowercase(),
            address: address.to_owned(),
            options,
        })
    }
}

/// A parsed codec specification: `raw`, `sbf`, `dbc:path/to/file.dbc`.
#[derive(Clone, Debug, PartialEq)]
pub struct CodecSpec {
    pub name: String,
    pub argument: Option<String>,
    pub options: Options,
}

impl CodecSpec {
    pub fn parse(spec: &str, options: Options) -> Result<Self> {
        let (name, argument) = match spec.split_once(':') {
            Some((n, a)) => (n, Some(a.to_owned())),
            None => (spec, None),
        };
        if name.is_empty() {
            return Err(Error::Config(format!("invalid codec {spec:?}")));
        }
        Ok(Self {
            name: name.to_ascii_lowercase(),
            argument,
            options,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_urls() {
        let e = Endpoint::parse("udp://10.0.0.2:5000?listen&bind=0.0.0.0:6000").unwrap();
        assert_eq!(
            (e.scheme.as_str(), e.address.as_str()),
            ("udp", "10.0.0.2:5000")
        );
        let mut o = e.options;
        assert!(o.take_bool("listen", false).unwrap());
        assert_eq!(o.take_str("bind").unwrap().as_deref(), Some("0.0.0.0:6000"));
        assert!(o.finish("udp").is_ok());
        assert!(Endpoint::parse("vcan0").is_err());
        let e = Endpoint::parse("CAN://vcan0").unwrap();
        assert_eq!(e.scheme, "can");
        assert!(e.options.finish("can").is_ok());
    }

    #[test]
    fn unknown_options_fail() {
        let e = Endpoint::parse("mem://x?typo=1").unwrap();
        assert!(e.options.finish("mem").is_err());
    }

    #[test]
    fn parses_codec_specs() {
        let c = CodecSpec::parse("dbc:/a/b:c.dbc", Options::new()).unwrap();
        assert_eq!(
            (c.name.as_str(), c.argument.as_deref()),
            ("dbc", Some("/a/b:c.dbc"))
        );
        let c = CodecSpec::parse("SBF", Options::new()).unwrap();
        assert_eq!((c.name.as_str(), c.argument), ("sbf", None));
    }
}
