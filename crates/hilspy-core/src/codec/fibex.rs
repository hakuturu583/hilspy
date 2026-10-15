//! `fibex:FILE`: CAN / CAN FD frames <-> the frames of one channel of an ASAM
//! FIBEX network description (see [`crate::fibex`]).
//!
//! Option `channel` selects the channel by short name or ID (optional when
//! the file has a single channel). The frames become a DBC database, so the
//! other options are the `dbc` codec's: `strict`, `scaling`,
//! `decode_choices`, `e2e` and `ecu` (see [`super::dbc`]; a frame's
//! transmitters and receivers are the ECUs whose connectors have an output or
//! input port for its frame triggering). Messages are named by
//! frame; their fields are the signals of all the frame's PDUs, as
//! `PDU.Signal` where two PDUs have a signal of the same name. Frames whose
//! identifier the channel lacks decode as nameless messages. A frame
//! triggering's `CYCLIC-TIMING` is the message's [`Codec::cycle_time`](super::Codec::cycle_time).

use super::dbc::{Dbc, SignalOptions};
use crate::error::{Error, Result};
use crate::url::CodecSpec;

pub(crate) fn open(spec: CodecSpec) -> Result<Dbc> {
    let path = spec.argument.ok_or_else(|| {
        Error::Config("the fibex codec needs a file: \"fibex:path/to/network.xml\"".into())
    })?;
    let mut o = spec.options;
    let channel = o.take_str("channel")?;
    let options = SignalOptions::take(&mut o)?;
    o.finish("the fibex codec")?;
    Dbc::new(crate::fibex::load(path, channel.as_deref())?, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Codec;
    use crate::message::{CanHeader, Message, Packet};
    use crate::value::{Options, Value};

    const NETWORK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/network.xml");

    fn codec(options: &[(&str, Value)]) -> Result<Dbc> {
        let mut o = Options::new();
        for (k, v) in options {
            o.insert(*k, v.clone());
        }
        open(CodecSpec::parse(&format!("fibex:{NETWORK}"), o)?)
    }

    fn body(options: &[(&str, Value)]) -> Dbc {
        let mut o = vec![("channel", Value::Str("Body".into()))];
        o.extend_from_slice(options);
        codec(&o).unwrap()
    }

    fn fields(kv: &[(&str, Value)]) -> Vec<(String, Value)> {
        kv.iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect()
    }

    fn round_trip(c: &mut Dbc, name: &str, kv: &[(&str, Value)]) -> (Packet, Message) {
        let mut packets = c.encode(&Message::new(name, fields(kv))).unwrap();
        assert_eq!(packets.len(), 1);
        let p = packets.remove(0);
        let m = c.decode(p.clone()).unwrap().remove(0);
        assert_eq!(m.name.as_deref(), Some(name));
        (p, m)
    }

    #[test]
    fn encode_decode_round_trips() {
        let mut c = body(&[]);
        let status = [
            ("Speed", Value::Float(123.45)),
            ("Gear", Value::Str("D".into())),
            ("BrakePressure", Value::Float(12.5)),
            ("BrakeActive", Value::Int(1)),
        ];
        let (p, m) = round_trip(&mut c, "VehicleStatus", &status);
        let h = p.can.unwrap();
        assert_eq!((h.id, h.extended, h.fd), (0x100, false, false));
        assert_eq!(p.data, [0x39, 0x30, 0x03, 0x00, 0x07, 0xD1, 0x00, 0x00]);
        assert_eq!(m.field("Gear"), Some(&Value::Str("D".into())));
        assert_eq!(m.field("BrakeActive"), Some(&Value::Int(1)));
        let steer = [
            ("SteeringAngle", Value::Float(-12.3)),
            ("Torque", Value::Float(-1.5)),
            ("Enable", Value::Int(0)),
        ];
        let (p, m) = round_trip(&mut c, "SteeringCmd", &steer);
        let h = p.can.unwrap();
        assert_eq!((h.id, h.extended, h.fd), (0x18FF_1234, true, false));
        assert_eq!(m.field("Torque"), Some(&Value::Float(-1.5)));
        let radar = [
            ("ObjectCount", Value::Int(3)),
            ("Distance", Value::Float(42.0)),
            ("RadarStatus", Value::Str("Blocked".into())),
        ];
        let (p, m) = round_trip(&mut c, "RadarObjects", &radar);
        let h = p.can.unwrap();
        assert!(h.fd && h.brs && p.data.len() == 16);
        assert_eq!(p.data[15], 0x80);
        assert_eq!(m.field("ObjectCount"), Some(&Value::Int(3)));
        assert_eq!(m.field("RadarStatus"), Some(&Value::Str("Blocked".into())));
    }

    #[test]
    fn qualified_names_on_the_chassis_channel() {
        let mut c = codec(&[("channel", Value::Str("CH_CHASSIS".into()))]).unwrap();
        let wheels = [
            ("Front.Left", Value::Float(1.0)),
            ("Front.Right", Value::Float(2.0)),
            ("Rear.Left", Value::Float(3.0)),
            ("Rear.Right", Value::Float(4.0)),
        ];
        let (p, m) = round_trip(&mut c, "WheelSpeeds", &wheels);
        assert_eq!(p.can.unwrap().id, 0x300);
        assert_eq!(p.data, [100, 0, 200, 0, 0x2C, 1, 0x90, 1]);
        assert_eq!(m.field("Rear.Right"), Some(&Value::Float(4.0)));
        let e = c
            .encode(&Message::new(
                "WheelSpeeds",
                fields(&[("Left", Value::Int(1))]),
            ))
            .unwrap_err();
        assert!(e.to_string().contains("no signal \"Left\""), "{e}");
    }

    #[test]
    fn unknown_frames_decode_as_nameless() {
        let mut c = body(&[]);
        // 0x300 is on the other channel only; 0x100 extended is not 0x100.
        for (id, extended) in [(0x300, false), (0x100, true)] {
            let p = Packet {
                data: vec![1, 2],
                can: Some(CanHeader {
                    id,
                    extended,
                    fd: false,
                    brs: false,
                }),
                timestamp: None,
                source: None,
            };
            let m = c.decode(p).unwrap().remove(0);
            assert_eq!(m.name, None);
            assert_eq!(m.packet.unwrap().data, [1, 2]);
        }
    }

    #[test]
    fn dbc_options_apply() {
        let status = [
            ("Speed", Value::Int(12345)),
            ("Gear", Value::Int(2)),
            ("BrakePressure", Value::Int(0)),
            ("BrakeActive", Value::Int(0)),
        ];
        let mut raw = body(&[
            ("scaling", Value::Bool(false)),
            ("decode_choices", Value::Bool(false)),
        ]);
        let (p, m) = round_trip(&mut raw, "VehicleStatus", &status);
        assert_eq!(&p.data[..3], [0x39, 0x30, 0x02]);
        assert_eq!(m.field("Speed"), Some(&Value::Int(12345)));
        assert_eq!(m.field("Gear"), Some(&Value::Int(2)));
        // Strict: every signal, within the physical constraints.
        let mut strict = body(&[]);
        let missing = Message::new("VehicleStatus", fields(&[("Speed", Value::Float(1.0))]));
        let e = strict.encode(&missing).unwrap_err().to_string();
        assert!(e.contains("missing value"), "{e}");
        let mut high = status.to_vec();
        high[0] = ("Speed", Value::Float(700.0));
        let e = strict
            .encode(&Message::new("VehicleStatus", fields(&high)))
            .unwrap_err()
            .to_string();
        assert!(e.contains("maximum"), "{e}");
        let mut lax = body(&[("strict", Value::Bool(false))]);
        let p = lax.encode(&missing).unwrap().remove(0);
        assert_eq!(p.data, [0x64, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn e2e_protects_fibex_frames() {
        let e2e = Value::Map(vec![(
            "VehicleStatus".into(),
            Value::Map(vec![
                ("profile".into(), Value::Int(11)),
                ("data_id".into(), Value::Int(0x42)),
                ("signals".into(), Value::List(vec![])),
            ]),
        )]);
        let mut tx = body(&[("e2e", e2e.clone())]);
        let mut rx = body(&[("e2e", e2e)]);
        let status = [
            ("Speed", Value::Float(1.0)),
            ("Gear", Value::Str("P".into())),
            ("BrakePressure", Value::Float(0.0)),
            ("BrakeActive", Value::Int(0)),
        ];
        let p = tx
            .encode(&Message::new("VehicleStatus", fields(&status)))
            .unwrap()
            .remove(0);
        let m = rx.decode(p).unwrap().remove(0);
        assert_eq!(m.field("e2e"), Some(&Value::Str("Ok".into())));
    }

    #[test]
    fn acts_as_an_ecu() {
        let gateway = || body(&[("ecu", Value::Str("Gateway".into()))]);
        let mut adas = body(&[("ecu", Value::Str("Adas".into()))]);
        let mut any = body(&[]);
        let status = [
            ("Speed", Value::Float(1.0)),
            ("Gear", Value::Str("D".into())),
            ("BrakePressure", Value::Float(0.0)),
            ("BrakeActive", Value::Int(0)),
        ];
        let steer = [
            ("SteeringAngle", Value::Float(1.0)),
            ("Torque", Value::Float(0.5)),
            ("Enable", Value::Int(1)),
        ];
        let radar = [
            ("ObjectCount", Value::Int(1)),
            ("Distance", Value::Float(2.0)),
            ("RadarStatus", Value::Str("Ok".into())),
        ];
        // Sending: only what the ECU transmits; the error names the sender.
        let mut gw = gateway();
        let p_status = gw
            .encode(&Message::new("VehicleStatus", fields(&status)))
            .unwrap()
            .remove(0);
        let e = gw
            .encode(&Message::new("SteeringCmd", fields(&steer)))
            .unwrap_err();
        assert!(matches!(e, Error::Encode(_)), "{e:?}");
        assert!(
            e.to_string()
                .contains("SteeringCmd is sent by Adas, not Gateway"),
            "{e}"
        );
        // Receiving: only what the ECU receives is decoded.
        let p_steer = any
            .encode(&Message::new("SteeringCmd", fields(&steer)))
            .unwrap()
            .remove(0);
        let p_radar = any
            .encode(&Message::new("RadarObjects", fields(&radar)))
            .unwrap()
            .remove(0);
        let name = |c: &mut Dbc, p: &Packet| c.decode(p.clone()).unwrap().remove(0).name;
        assert_eq!(name(&mut gw, &p_steer).as_deref(), Some("SteeringCmd"));
        assert_eq!(name(&mut gw, &p_radar), None);
        assert_eq!(name(&mut gw, &p_status), None, "its own frame");
        assert_eq!(name(&mut adas, &p_status).as_deref(), Some("VehicleStatus"));
        assert_eq!(name(&mut adas, &p_radar).as_deref(), Some("RadarObjects"));
        let nameless = gw.decode(p_radar.clone()).unwrap().remove(0);
        assert!(nameless.fields.is_empty());
        assert_eq!(nameless.packet.unwrap().data, p_radar.data);
        // Without the option, everything as before.
        assert_eq!(name(&mut any, &p_radar).as_deref(), Some("RadarObjects"));
        // An ECU on the other channel only sends and receives nothing here.
        let mut abs = body(&[("ecu", Value::Str("Abs".into()))]);
        let e = abs
            .encode(&Message::new("VehicleStatus", fields(&status)))
            .unwrap_err();
        assert!(e.to_string().contains("sent by Gateway, not Abs"), "{e}");
        assert_eq!(name(&mut abs, &p_status), None);
    }

    #[test]
    fn option_errors() {
        let err = |o: &[(&str, Value)]| codec(o).err().unwrap().to_string();
        let e = err(&[]);
        assert!(
            e.contains("channel option") && e.contains("Chassis (CH_CHASSIS)"),
            "{e}"
        );
        let e = err(&[("channel", Value::Str("Nope".into()))]);
        assert!(e.contains("\"Nope\""), "{e}");
        let e = err(&[("channel", Value::Bool(true))]);
        assert!(e.contains("not a string"), "{e}");
        let e = err(&[
            ("channel", Value::Str("Body".into())),
            ("typo", Value::Int(1)),
        ]);
        assert!(
            e.contains("the fibex codec does not take") && e.contains("typo"),
            "{e}"
        );
        let e = err(&[
            ("channel", Value::Str("Body".into())),
            ("strict", Value::Str("maybe".into())),
        ]);
        assert!(e.contains("not a boolean"), "{e}");
        let e = err(&[
            ("channel", Value::Str("Body".into())),
            (
                "e2e",
                Value::Map(vec![("WheelSpeeds".into(), Value::Map(vec![]))]),
            ),
        ]);
        assert!(e.contains("WheelSpeeds"), "{e}");
        let e = err(&[
            ("channel", Value::Str("Body".into())),
            ("ecu", Value::Str("Engine".into())),
        ]);
        assert!(
            e.contains("option ecu: no ECU \"Engine\" (ECUs: Gateway, Adas, Radar, Abs)"),
            "{e}"
        );
        let e = err(&[
            ("channel", Value::Str("Body".into())),
            ("ecu", Value::Bool(true)),
        ]);
        assert!(e.contains("not a string"), "{e}");
        let no_file = open(CodecSpec::parse("fibex", Options::new()).unwrap());
        assert!(no_file.err().unwrap().to_string().contains("needs a file"));
        assert!(crate::codec::names().contains(&"fibex".to_string()));
    }
}
