//! `rtcm`: RTCM 3 frames (GNSS corrections) <-> messages.
//!
//! Decoding frames the byte stream (one decoder per packet source), checks
//! the CRC and parses each message with the `rtcm-rs` crate: a message
//! named after its number (`1005`, `1077`, ...) with the crate's fields.
//! Encoding goes the other way for any message the crate knows, from the
//! complete field set decoding reports (a received correction stream can be
//! forwarded or edited and re-sent).

use std::collections::HashMap;

use rtcm_rs::{next_msg_frame, Message as RtcmMessage, MessageBuilder};

use super::Codec;
use crate::error::{Error, Result};
use crate::message::{Message, Packet};
use crate::url::CodecSpec;
use crate::value::{from_json, to_json, Value};

/// Longest buffer kept while waiting for the rest of a frame (a frame holds
/// at most 1029 bytes).
const MAX_BUFFER: usize = 1 << 16;

pub(crate) struct Rtcm {
    buffers: HashMap<Option<u64>, Vec<u8>>,
    builder: MessageBuilder,
}

impl Rtcm {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        if spec.argument.is_some() {
            return Err(Error::Config("the rtcm codec takes no argument".into()));
        }
        spec.options.finish("the rtcm codec")?;
        Ok(Self {
            buffers: HashMap::new(),
            builder: MessageBuilder::new(),
        })
    }
}

fn message(
    frame: &rtcm_rs::MessageFrame<'_>,
    timestamp: Option<f64>,
    source: Option<u64>,
) -> Message {
    let number = frame.message_number();
    let parsed = serde_json::to_value(frame.get_message()).ok();
    let (name, fields) = match parsed {
        Some(serde_json::Value::Object(mut m)) if m.len() == 1 => {
            let (key, body) = m.iter_mut().next().expect("one entry");
            let fields = match from_json(body.take()) {
                Value::Map(f) => f,
                _ => Vec::new(),
            };
            (key.trim_start_matches("Msg").to_owned(), fields)
        }
        // Messages the crate does not parse.
        _ => (
            number.map_or_else(|| "RTCM".into(), |n| n.to_string()),
            Vec::new(),
        ),
    };
    Message {
        name: Some(name),
        fields,
        packet: Some(Packet {
            data: frame.frame_data().to_vec(),
            can: None,
            timestamp,
            source,
        }),
    }
}

impl Codec for Rtcm {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let name = message.name.as_deref().unwrap_or_default();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::NotFound(format!(
                "the rtcm codec encodes messages named by number (1005, 1077, ...), not {name:?}"
            )));
        }
        let body = to_json(&Value::Map(message.fields.clone()));
        let json = serde_json::json!({ format!("Msg{name}"): body });
        let parsed: RtcmMessage =
            serde_json::from_value(json).map_err(|e| Error::encode(format!("RTCM {name}: {e}")))?;
        let data = self
            .builder
            .build_message(&parsed)
            .map_err(|e| Error::encode(format!("RTCM {name}: {e:?}")))?
            .to_vec();
        Ok(vec![Packet::bytes(data)])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let buf = self.buffers.entry(packet.source).or_default();
        buf.extend_from_slice(&packet.data);
        let mut out = Vec::new();
        let mut used = 0;
        loop {
            let (consumed, frame) = next_msg_frame(&buf[used..]);
            let Some(frame) = frame else {
                used += consumed;
                break;
            };
            out.push(message(&frame, packet.timestamp, packet.source));
            used += consumed;
        }
        buf.drain(..used);
        if buf.len() > MAX_BUFFER {
            buf.clear();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> Rtcm {
        Rtcm::open(CodecSpec::parse("rtcm", Default::default()).unwrap()).unwrap()
    }

    #[test]
    fn station_message_round_trips() {
        let mut c = codec();
        let m = Message::new(
            "1005",
            vec![
                ("reference_station_id".into(), Value::Int(2003)),
                ("reserved_24_6".into(), Value::Int(0)),
                ("gps_flag".into(), Value::Int(1)),
                ("glonass_flag".into(), Value::Int(1)),
                ("galileo_flag".into(), Value::Int(0)),
                ("reference_station_ind".into(), Value::Int(0)),
                (
                    "antenna_ref_point_ecef_x_m".into(),
                    Value::Float(-3_957_199.234_5),
                ),
                ("single_receiver_osc_ind".into(), Value::Int(0)),
                ("reserved_73_1".into(), Value::Int(0)),
                (
                    "antenna_ref_point_ecef_y_m".into(),
                    Value::Float(3_310_199.123_4),
                ),
                ("quarter_cycle_ind".into(), Value::Int(0)),
                (
                    "antenna_ref_point_ecef_z_m".into(),
                    Value::Float(3_737_711.876_5),
                ),
            ],
        );
        let bytes = c.encode(&m).unwrap().remove(0).data;
        assert_eq!(bytes[0], 0xD3);
        let mut stream = b"junk".to_vec();
        stream.extend_from_slice(&bytes);
        stream.extend_from_slice(&bytes);
        let mut out = Vec::new();
        for chunk in stream.chunks(5) {
            out.extend(c.decode(Packet::bytes(chunk.to_vec())).unwrap());
        }
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name.as_deref(), Some("1005"));
        assert_eq!(
            out[0].field("reference_station_id"),
            Some(&Value::Int(2003))
        );
        let x = out[0]
            .field("antenna_ref_point_ecef_x_m")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((x + 3_957_199.234_5).abs() < 1e-4);
        assert_eq!(out[0].packet.as_ref().unwrap().data, bytes);
        // A decoded message encodes back to the same frame.
        assert_eq!(c.encode(&out[1]).unwrap().remove(0).data, bytes);
        assert!(c.encode(&Message::new("1005", vec![])).is_err());

        assert!(c.encode(&Message::new("GGA", vec![])).is_err());
    }

    #[test]
    fn generated_messages_round_trip() {
        use rand::SeedableRng;
        let mut c = codec();
        let rng = || rand::rngs::StdRng::seed_from_u64(7);
        let mut gen = rtcm_rs::val_gen::ValGen::new(rng(), rng(), rng());
        // Station, antenna, ephemerides and MSM4/5/7 observations of each
        // constellation: fields, nested segments and satellite lists.
        let numbers = [
            1005, 1006, 1008, 1019, 1020, 1033, 1042, 1045, 1046, 1074, 1075, 1077, 1084, 1087,
            1094, 1097, 1124, 1127, 1230,
        ];
        for n in numbers {
            for _ in 0..5 {
                let bytes = MessageBuilder::new()
                    .build_generated_message(&mut gen, n)
                    .unwrap()
                    .to_vec();
                let d = c.decode(Packet::bytes(bytes.clone())).unwrap().remove(0);
                assert_eq!(d.name.as_deref(), Some(n.to_string().as_str()));
                assert!(!d.fields.is_empty(), "{n}");
                let again = c.encode(&d).unwrap_or_else(|e| panic!("{n}: {e}"));
                assert_eq!(again[0].data, bytes, "{n}");
            }
        }
    }
}
