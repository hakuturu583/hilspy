//! `raw`: no decoding; every packet is a nameless message.

use super::Codec;
use crate::error::{Error, Result};
use crate::message::{Message, Packet};
use crate::url::CodecSpec;

pub(crate) struct Raw;

impl Raw {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        if spec.argument.is_some() {
            return Err(Error::Config("the raw codec takes no argument".into()));
        }
        spec.options.finish("the raw codec")?;
        Ok(Raw)
    }
}

impl Codec for Raw {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        Err(Error::encode(format!(
            "the raw codec has no named messages ({:?}); send data instead",
            message.name.as_deref().unwrap_or("")
        )))
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        Ok(vec![Message::raw(packet)])
    }
}
