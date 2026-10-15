//! `fibex:FILE`: CAN / CAN FD, LIN and FlexRay frames <-> the frames of one
//! channel of an ASAM FIBEX network description (see [`crate::fibex`]).
//!
//! Option `channel` selects the channel by short name or ID (optional when
//! the file has a single channel). The frames become a DBC database, so the
//! other options are the `dbc` codec's: `strict`, `scaling`,
//! `decode_choices`, `e2e` and `ecu` (see [`super::dbc`]; a frame's
//! transmitters and receivers are the ECUs whose connectors have an output or
//! input port for its frame triggering). Messages are named by
//! frame; their fields are the signals of all the frame's PDUs, as
//! `PDU.Signal` where two PDUs have a signal of the same name. Frames the
//! channel lacks decode as nameless messages. A frame
//! triggering's `CYCLIC-TIMING` is the message's [`Codec::cycle_time`](super::Codec::cycle_time).
//!
//! On a LIN channel (its cluster's `PROTOCOL` is `LIN`) packets are LIN
//! frames: encoding sets the frame identifier, the checksum model and the
//! checksum; decoding finds the frame by identifier and verifies the checksum
//! under the frame's model (whatever model the header claims), and a
//! mismatch is a decode error naming the frame and both checksums. A frame
//! with a bad checksum has no trustworthy payload, so, like a CAN controller
//! that drops a frame failing its CRC (and unlike an E2E check, which judges
//! an intact payload and becomes the `e2e` field), it yields no message;
//! `recv` raises and the next frame decodes as usual. The schedule tables give
//! cycle times (option `schedule` picks the table, see [`crate::fibex`]).
//!
//! On a FlexRay channel packets are FlexRay frames on the channel's channel
//! (`A`, `B`, or `AB` when the file names none). Decoding takes a frame seen
//! on that channel (`AB` overlaps both) and finds the frame sent in its slot
//! in its cycle (`cycle % repetition == base cycle`); any other frame,
//! including one in the right slot of a cycle the frame is not sent in, is
//! nameless. Encoding pads the payload to whole two-byte words and picks the
//! cycle counter from the message's time (the channel's clock, else the wall
//! clock): the first cycle at or after `floor(time / cycle length)` modulo 64
//! in which the frame is sent (the frame's first base cycle when the cluster
//! has no cycle length). The cycle time is the cycle length times the
//! repetition, so `every(None, ...)` on a clock sends each frame in its own
//! cycles.

use std::collections::HashMap;
use std::time::Duration;

use super::dbc::{Dbc, SignalOptions};
use super::Codec;
use crate::error::{Error, Result};
use crate::fibex::{Bus, FlexRayFrame, LinFrame};
use crate::message::{now_timestamp, FlexRayChannel, Message, Packet};
use crate::url::CodecSpec;

pub(crate) fn open(spec: CodecSpec) -> Result<Box<dyn Codec>> {
    let path = spec.argument.ok_or_else(|| {
        Error::Config("the fibex codec needs a file: \"fibex:path/to/network.xml\"".into())
    })?;
    let mut o = spec.options;
    let channel = o.take_str("channel")?;
    let schedule = o.take_str("schedule")?;
    let options = SignalOptions::take(&mut o)?;
    o.finish("the fibex codec")?;
    let description = crate::fibex::load_channel(path, channel.as_deref(), schedule.as_deref())?;
    let dbc = Dbc::new(description.database, options)?;
    Ok(match description.bus {
        Bus::Can => Box::new(dbc),
        Bus::Lin(lin) => Box::new(LinCodec {
            dbc,
            by_name: lin
                .frames
                .iter()
                .map(|f| (f.name.clone(), f.clone()))
                .collect(),
            by_id: lin.frames.into_iter().map(|f| (f.id, f)).collect(),
        }),
        Bus::FlexRay(fr) => Box::new(FlexRayCodec {
            dbc,
            cycle: fr.cycle,
            channel: fr.channel,
            frames: fr.frames,
        }),
    })
}

/// The `dbc` codec over a LIN channel's frames, with LIN framing.
struct LinCodec {
    dbc: Dbc,
    by_name: HashMap<String, LinFrame>,
    by_id: HashMap<u8, LinFrame>,
}

impl Codec for LinCodec {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let data = self.dbc.payload(message)?;
        let name = message.name.as_deref().unwrap_or_default();
        let f = self
            .by_name
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("no message named {name:?}")))?;
        Ok(vec![Packet::lin(f.id, f.checksum, data)])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let Some(f) = packet.lin.and_then(|h| self.by_id.get(&h.id)) else {
            return Ok(vec![Message::raw(packet)]);
        };
        let f = f.clone();
        let m = self.dbc.decode_as(&f.name, packet, |p| {
            let got = p.lin.map(|h| h.checksum).unwrap_or_default();
            let want = f.checksum.compute(f.id, &p.data);
            if got == want {
                Ok(())
            } else {
                Err(Error::decode(format!(
                    "LIN frame {} (id 0x{:02X}): checksum 0x{got:02X}, expected the {} \
                     checksum 0x{want:02X}",
                    f.name,
                    f.id,
                    match f.checksum {
                        crate::message::LinChecksum::Classic => "classic",
                        crate::message::LinChecksum::Enhanced => "enhanced",
                    }
                )))
            }
        })?;
        Ok(vec![m])
    }

    fn cycle_time(&self, message: &Message) -> Option<Duration> {
        self.dbc.cycle_time(message)
    }
}

/// The `dbc` codec over a FlexRay channel's frames, with FlexRay framing.
struct FlexRayCodec {
    dbc: Dbc,
    cycle: Option<Duration>,
    channel: FlexRayChannel,
    frames: Vec<FlexRayFrame>,
}

impl FlexRayCodec {
    /// The slot and cycle counter to send `frame` in at `time` (seconds).
    fn schedule(&self, frame: &FlexRayFrame, time: f64) -> (u16, u8) {
        let first = frame.timings[0];
        let Some(cycle) = self.cycle else {
            return (first.slot, first.base_cycle);
        };
        // A tolerance for times at a cycle boundary computed in floats.
        let now = (time / cycle.as_secs_f64() + 1e-6).floor().max(0.0) as u64;
        (0..64u64)
            .find_map(|k| {
                let c = ((now + k) % 64) as u8;
                frame
                    .timings
                    .iter()
                    .find(|t| c % t.repetition == t.base_cycle)
                    .map(|t| (t.slot, c))
            })
            .unwrap_or((first.slot, first.base_cycle))
    }
}

impl Codec for FlexRayCodec {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let mut data = self.dbc.payload(message)?;
        if data.len() % 2 == 1 {
            data.push(0);
        }
        let name = message.name.as_deref().unwrap_or_default();
        let f = self
            .frames
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| Error::NotFound(format!("no message named {name:?}")))?;
        let time = message.timestamp().unwrap_or_else(now_timestamp);
        let (slot, cycle) = self.schedule(f, time);
        Ok(vec![Packet::flexray(slot, cycle, self.channel, data)])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let frame = packet.flexray.and_then(|h| {
            if !h.channel.overlaps(self.channel) {
                return None;
            }
            self.frames.iter().find(|f| f.matches(h.slot, h.cycle))
        });
        let Some(f) = frame else {
            return Ok(vec![Message::raw(packet)]);
        };
        let name = f.name.clone();
        Ok(vec![self.dbc.decode_as(&name, packet, |_| Ok(()))?])
    }

    fn cycle_time(&self, message: &Message) -> Option<Duration> {
        self.dbc.cycle_time(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Codec;
    use crate::message::{CanHeader, FlexRayChannel, LinChecksum, Message, Packet};
    use crate::value::{Options, Value};

    const NETWORK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/network.xml");

    fn codec(options: &[(&str, Value)]) -> Result<Box<dyn Codec>> {
        let mut o = Options::new();
        for (k, v) in options {
            o.insert(*k, v.clone());
        }
        open(CodecSpec::parse(&format!("fibex:{NETWORK}"), o)?)
    }

    fn body(options: &[(&str, Value)]) -> Box<dyn Codec> {
        let mut o = vec![("channel", Value::Str("Body".into()))];
        o.extend_from_slice(options);
        codec(&o).unwrap()
    }

    fn fields(kv: &[(&str, Value)]) -> Vec<(String, Value)> {
        kv.iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect()
    }

    fn round_trip(c: &mut Box<dyn Codec>, name: &str, kv: &[(&str, Value)]) -> (Packet, Message) {
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
                lin: None,
                flexray: None,
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
        let name = |c: &mut Box<dyn Codec>, p: &Packet| c.decode(p.clone()).unwrap().remove(0).name;
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

    const LIN_FLEXRAY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/data/lin_flexray.xml"
    );

    fn lin_flexray(channel: &str, options: &[(&str, Value)]) -> Result<Box<dyn Codec>> {
        let mut o = Options::new();
        o.insert("channel", Value::Str(channel.into()));
        for (k, v) in options {
            o.insert(*k, v.clone());
        }
        open(CodecSpec::parse(&format!("fibex:{LIN_FLEXRAY}"), o)?)
    }

    fn light_cmd() -> Vec<(String, Value)> {
        fields(&[
            ("Brightness", Value::Float(50.0)),
            ("Mode", Value::Str("High".into())),
        ])
    }

    #[test]
    fn lin_frames_carry_identifier_and_checksum() {
        let mut c = lin_flexray("Lin", &[]).unwrap();
        // Enhanced: !(pid 0x50 + 0x64 + 0x02) = !0xB6.
        let (p, m) = round_trip(
            &mut c,
            "LightCmd",
            &[
                ("Brightness", Value::Float(50.0)),
                ("Mode", Value::Str("High".into())),
            ],
        );
        assert_eq!(p.data, [0x64, 0x02]);
        assert!(p.can.is_none());
        let h = p.lin.unwrap();
        assert_eq!(
            (h.id, h.protected_id(), h.checksum_model, h.checksum),
            (0x10, 0x50, LinChecksum::Enhanced, 0x49)
        );
        assert_eq!(m.field("Mode"), Some(&Value::Str("High".into())));
        // Classic (CHECKSUM-TYPE): !0x0F.
        let (p, m) = round_trip(
            &mut c,
            "LightStatus",
            &[("Current", Value::Float(1.5)), ("Fault", Value::Int(0))],
        );
        let h = p.lin.unwrap();
        assert_eq!(
            (h.id, h.checksum_model, h.checksum),
            (0x11, LinChecksum::Classic, 0xF0)
        );
        assert_eq!(m.field("Current"), Some(&Value::Float(1.5)));
        // Diagnostic: classic, !(0x7F + 0x06 + 0xB2 with carry 0x38).
        let (p, _) = round_trip(
            &mut c,
            "MasterReq",
            &[
                ("Nad", Value::Int(0x7F)),
                ("Pci", Value::Int(0x06)),
                ("Sid", Value::Int(0xB2)),
            ],
        );
        assert_eq!(p.data, [0x7F, 0x06, 0xB2, 0, 0, 0, 0, 0]);
        assert_eq!(p.lin.unwrap().checksum, 0xC7);
        // Cycle times from the schedule tables.
        let cycle = |c: &dyn Codec, n: &str| c.cycle_time(&Message::new(n, vec![]));
        assert_eq!(
            cycle(c.as_ref(), "LightCmd"),
            Some(Duration::from_millis(20))
        );
        assert_eq!(
            cycle(c.as_ref(), "LightStatus"),
            Some(Duration::from_millis(40))
        );
        let diag = lin_flexray("Lin", &[("schedule", Value::Str("Diagnostic".into()))]).unwrap();
        assert_eq!(cycle(diag.as_ref(), "LightCmd"), None);
        assert_eq!(
            cycle(diag.as_ref(), "MasterReq"),
            Some(Duration::from_millis(20))
        );
    }

    #[test]
    fn lin_checksums_are_verified() {
        let mut c = lin_flexray("Lin", &[]).unwrap();
        let good = c
            .encode(&Message::new("LightCmd", light_cmd()))
            .unwrap()
            .remove(0);
        let mut bad = good.clone();
        bad.lin.as_mut().unwrap().checksum ^= 0x01;
        let e = c.decode(bad.clone()).unwrap_err();
        assert!(matches!(e, Error::Decode(_)), "{e:?}");
        assert!(
            e.to_string().contains(
                "LIN frame LightCmd (id 0x10): checksum 0x48, expected the enhanced checksum 0x49"
            ),
            "{e}"
        );
        // The receiver checks the frame's model, not the header's: a classic
        // checksum on an enhanced frame is wrong.
        let classic = Packet::lin(0x10, LinChecksum::Classic, good.data.clone());
        assert!(c.decode(classic).is_err());
        // Next frames decode as usual; unknown ids and other buses are nameless.
        assert_eq!(
            c.decode(good.clone()).unwrap()[0].name.as_deref(),
            Some("LightCmd")
        );
        for p in [
            Packet::lin(0x20, LinChecksum::Enhanced, vec![1]),
            Packet::can(0x10, good.data.clone()),
            Packet::bytes(good.data.clone()),
        ] {
            assert_eq!(c.decode(p).unwrap()[0].name, None);
        }
        // An ECU decodes (and so checks) only what it receives.
        let mut bcm = lin_flexray("Lin", &[("ecu", Value::Str("Bcm".into()))]).unwrap();
        assert_eq!(bcm.decode(bad).unwrap()[0].name, None, "its own frame");
        let mut lamp = lin_flexray("Lin", &[("ecu", Value::Str("Lamp".into()))]).unwrap();
        let e = lamp
            .encode(&Message::new("LightCmd", light_cmd()))
            .unwrap_err();
        assert!(e.to_string().contains("sent by Bcm, not Lamp"), "{e}");
        assert_eq!(
            lamp.decode(good).unwrap()[0].name.as_deref(),
            Some("LightCmd")
        );
        // The schedule option is LIN only.
        let e = codec(&[
            ("channel", Value::Str("Body".into())),
            ("schedule", Value::Str("Normal".into())),
        ])
        .err()
        .unwrap();
        assert!(e.to_string().contains("not a LIN channel"), "{e}");
    }

    fn at(name: &str, kv: &[(&str, Value)], time: f64) -> Message {
        let mut m = Message::new(name, fields(kv));
        m.packet = Some(Packet {
            timestamp: Some(time),
            ..Packet::default()
        });
        m
    }

    #[test]
    fn flexray_frames_carry_slot_cycle_and_channel() {
        let mut a = lin_flexray("FlexRayA", &[]).unwrap();
        let brake = [
            ("BrakeTorque", Value::Float(100.0)),
            ("Counter", Value::Int(1)),
        ];
        let yaw = [("YawRate", Value::Float(1.5)), ("Valid", Value::Int(1))];
        let diag = [("Code", Value::Int(0x42)), ("State", Value::Int(2))];
        let header = |c: &mut Box<dyn Codec>, m: Message| {
            let p = c.encode(&m).unwrap().remove(0);
            let h = p.flexray.unwrap();
            (h.slot, h.cycle, h.channel, p)
        };
        // Cycle length 5 ms: the cycle counter is the time's cycle, or the
        // next one the frame is sent in.
        let (slot, cycle, channel, p) = header(&mut a, at("BrakeFr", &brake, 0.012));
        assert_eq!((slot, cycle, channel), (3, 2, FlexRayChannel::A));
        assert_eq!(p.data, [200, 0, 1, 0]);
        assert!(p.can.is_none() && p.lin.is_none());
        let cycles: Vec<_> = [0.0, 0.005, 0.0051, 0.02, 0.3, 0.32]
            .iter()
            .map(|&t| header(&mut a, at("YawFr", &yaw, t)).1)
            .collect();
        // Base 1, repetition 4; 0.3 s is cycle 60, 0.32 s cycle 64 = 0.
        assert_eq!(cycles, [1, 1, 1, 5, 61, 1]);
        let (slot, cycle, _, p) = header(&mut a, at("DiagFr", &diag, 0.0));
        assert_eq!((slot, cycle), (7, 3));
        assert_eq!(p.data, [0x42, 2, 0, 0], "3 bytes padded to a word");
        // Decoding by slot and cycle.
        let name = |c: &mut Box<dyn Codec>, p: Packet| c.decode(p).unwrap().remove(0).name;
        let payload = vec![150, 0, 1, 0];
        let fr = |slot, cycle, channel| Packet::flexray(slot, cycle, channel, payload.clone());
        assert_eq!(
            name(&mut a, fr(7, 5, FlexRayChannel::A)).as_deref(),
            Some("YawFr")
        );
        assert_eq!(
            name(&mut a, fr(7, 7, FlexRayChannel::A)).as_deref(),
            Some("DiagFr")
        );
        assert_eq!(
            name(&mut a, fr(7, 2, FlexRayChannel::A)),
            None,
            "wrong cycle"
        );
        assert_eq!(
            name(&mut a, fr(8, 1, FlexRayChannel::A)),
            None,
            "unknown slot"
        );
        assert_eq!(
            name(&mut a, fr(7, 1, FlexRayChannel::B)),
            None,
            "other channel"
        );
        assert_eq!(
            name(&mut a, fr(7, 1, FlexRayChannel::AB)).as_deref(),
            Some("YawFr")
        );
        let m = a.decode(fr(7, 1, FlexRayChannel::A)).unwrap().remove(0);
        assert_eq!(m.field("YawRate"), Some(&Value::Float(1.5)));
        assert_eq!(name(&mut a, Packet::can(7, payload.clone())), None);
        let mut b = lin_flexray("FlexRayB", &[]).unwrap();
        assert_eq!(name(&mut b, fr(3, 9, FlexRayChannel::A)), None);
        assert_eq!(
            name(&mut b, fr(3, 9, FlexRayChannel::B)).as_deref(),
            Some("BrakeFr")
        );
        let (_, _, channel, _) = header(&mut b, at("BrakeFr", &brake, 0.0));
        assert_eq!(channel, FlexRayChannel::B);
        // Cycle times: cycle length times repetition.
        let cycle = |c: &dyn Codec, n: &str| c.cycle_time(&Message::new(n, vec![]));
        assert_eq!(cycle(a.as_ref(), "BrakeFr"), Some(Duration::from_millis(5)));
        assert_eq!(cycle(a.as_ref(), "YawFr"), Some(Duration::from_millis(20)));
        // The ecu option applies as on CAN.
        let mut esc = lin_flexray("FlexRayA", &[("ecu", Value::Str("Esc".into()))]).unwrap();
        let e = esc.encode(&at("YawFr", &yaw, 0.0)).unwrap_err();
        assert!(e.to_string().contains("sent by Bcm, not Esc"), "{e}");
        assert_eq!(
            name(&mut esc, fr(7, 1, FlexRayChannel::A)).as_deref(),
            Some("YawFr")
        );
        assert_eq!(
            name(&mut esc, fr(3, 1, FlexRayChannel::A)),
            None,
            "its own frame"
        );
    }
}
