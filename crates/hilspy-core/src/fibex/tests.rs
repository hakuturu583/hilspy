use std::path::PathBuf;

use super::*;
use crate::dbc::{DecodeOptions, EncodeOptions, Value};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

fn network() -> String {
    std::fs::read_to_string(data("network.xml")).unwrap()
}

/// A variant of the fixture written to a temporary file.
fn variant(name: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("hilspy-fibex-{}-{name}.xml", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

/// `text` without the first element that starts with `start`.
fn without(text: &str, start: &str, end: &str) -> String {
    let i = text.find(start).unwrap();
    let j = i + text[i..].find(end).unwrap() + end.len();
    format!("{}{}", &text[..i], &text[j..])
}

fn load_err(path: &Path, channel: Option<&str>) -> String {
    load(path, channel).unwrap_err().to_string()
}

#[test]
fn parses_frames_pdus_and_codings() {
    let db = load(data("network.xml"), Some("Body")).unwrap();
    let names: Vec<_> = db.messages.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["VehicleStatus", "SteeringCmd", "RadarObjects"]);

    let status = db.message_by_id(0x100, false).unwrap();
    assert_eq!((status.size, status.fd), (8, false));
    assert_eq!(status.comment.as_deref(), Some("Speed, gear and brake"));
    // Two PDUs; the second one's signals are moved by its 4-byte offset.
    let pressure = status.signal("BrakePressure").unwrap();
    assert_eq!(
        (pressure.start_bit, pressure.byte_order),
        (39, ByteOrder::BigEndian)
    );
    let speed = status.signal("Speed").unwrap();
    assert_eq!((speed.factor, speed.offset), (0.01, 0.0));
    assert_eq!((speed.minimum, speed.maximum), (Some(0.0), Some(655.35)));
    assert_eq!(speed.unit, "km/h");
    assert_eq!(speed.comment.as_deref(), Some("Vehicle speed"));
    assert_eq!(status.signal("Gear").unwrap().choices[3], (3, "D".into()));

    let steering = db.message_by_id(0x18FF_1234, true).unwrap();
    let torque = steering.signal("Torque").unwrap();
    assert!(torque.signed);
    assert_eq!(torque.factor, 0.05);
    assert_eq!(torque.choices, vec![(127, "Invalid".to_string())]);

    let radar = db.message_by_name("RadarObjects").unwrap();
    assert!(radar.fd && !radar.extended);
    // LINEAR with a denominator: 1 / 100.
    assert_eq!(radar.signal("Distance").unwrap().factor, 0.01);
}

#[test]
fn matches_the_equivalent_dbc() {
    let fibex = load(data("network.xml"), Some("CH_BODY")).unwrap();
    let dbc = Database::from_file(data("network.dbc")).unwrap();
    assert_eq!(fibex.messages.len(), dbc.messages.len());
    for d in &dbc.messages {
        let f = fibex.message_by_name(&d.name).unwrap();
        assert_eq!(
            (f.id, f.extended, f.size, f.fd, &f.comment),
            (d.id, d.extended, d.size, d.fd, &d.comment),
            "{}",
            d.name
        );
        let mut expected = d.signals.clone();
        for s in &mut expected {
            s.receivers.clear();
        }
        assert_eq!(f.signals, expected, "{}", d.name);
    }
}

#[test]
fn selects_a_channel() {
    let chassis = load(data("network.xml"), Some("Chassis")).unwrap();
    assert_eq!(chassis.messages.len(), 2);
    // The same frame, triggered with another identifier.
    assert_eq!(chassis.message_by_name("VehicleStatus").unwrap().id, 0x101);
    // Signals whose names clash across PDUs are qualified by PDU.
    let wheels = chassis.message_by_id(0x300, false).unwrap();
    let names: Vec<_> = wheels.signals.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        ["Front.Left", "Front.Right", "Rear.Left", "Rear.Right"]
    );
    assert_eq!(wheels.signals[2].start_bit, 32);
    // By ID too.
    let by_id = load(data("network.xml"), Some("CH_CHASSIS")).unwrap();
    assert_eq!(by_id.messages, chassis.messages);
}

#[test]
fn channel_errors_list_the_channels() {
    let path = data("network.xml");
    let e = load(&path, None).unwrap_err();
    assert!(matches!(e, Error::Config(_)), "{e:?}");
    let e = e.to_string();
    assert!(e.contains("2 channels"), "{e}");
    assert!(e.contains("Body (CH_BODY), Chassis (CH_CHASSIS)"), "{e}");
    let e = load_err(&path, Some("Powertrain"));
    assert!(
        e.contains("\"Powertrain\"") && e.contains("Body (CH_BODY)"),
        "{e}"
    );
}

#[test]
fn a_single_channel_needs_no_option() {
    let text = without(
        &network(),
        "<fx:CHANNEL ID=\"CH_CHASSIS\">",
        "</fx:CHANNEL>",
    );
    let path = variant("single", &text);
    let db = load(&path, None).unwrap();
    assert_eq!(db.messages.len(), 3);
    assert!(load_err(&path, Some("Chassis")).contains("Body (CH_BODY)"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn round_trips_through_the_dbc_model() {
    let db = load(data("network.xml"), Some("Body")).unwrap();
    let m = db.message_by_name("SteeringCmd").unwrap();
    let values = vec![
        ("SteeringAngle".to_string(), Value::Float(-90.5)),
        ("Torque".to_string(), Value::Str("Invalid".into())),
        ("Enable".to_string(), Value::Int(1)),
    ];
    let data = m.encode(&values, &EncodeOptions::default()).unwrap();
    assert_eq!(data, [0xFC, 0x77, 0x7F, 0x01, 0, 0, 0, 0]);
    let decoded = m.decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[1], ("Torque".into(), Value::Str("Invalid".into())));
    match decoded[0].1 {
        Value::Float(a) => assert!((a + 90.5).abs() < 1e-9),
        ref v => panic!("{v:?}"),
    }
}

#[test]
fn file_errors() {
    let e = load(data("nonexistent.xml"), None).unwrap_err();
    assert!(matches!(e, Error::Io(_)), "{e:?}");
    let e = load_err(&data("network.dbc"), None);
    assert!(e.contains("FIBEX error"), "{e}");
    let path = variant("root", "<?xml version=\"1.0\"?><NETWORK/>");
    assert!(load_err(&path, None).contains("not FIBEX"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unsupported_content_is_reported() {
    let text = network();
    let cases = [
        (
            "range",
            text.replace(
                "<ho:LOWER-LIMIT>3</ho:LOWER-LIMIT>\n                <ho:UPPER-LIMIT>3</ho:UPPER-LIMIT>",
                "<ho:LOWER-LIMIT>3</ho:LOWER-LIMIT>\n                <ho:UPPER-LIMIT>7</ho:UPPER-LIMIT>",
            ),
            "texttable ranges",
        ),
        (
            "category",
            text.replacen(
                "<ho:CATEGORY>IDENTICAL</ho:CATEGORY>",
                "<ho:CATEGORY>FORMULA</ho:CATEGORY>",
                1,
            ),
            "unsupported compu method",
        ),
        (
            "fit",
            text.replace(
                "<fx:BIT-POSITION>127</fx:BIT-POSITION>",
                "<fx:BIT-POSITION>128</fx:BIT-POSITION>",
            ),
            "do not fit",
        ),
        (
            "pdu",
            text.replace(
                "<fx:PDU-REF ID-REF=\"P_Brake\"/>\n            <fx:BIT-POSITION>32</fx:BIT-POSITION>",
                "<fx:PDU-REF ID-REF=\"P_Brake\"/>\n            <fx:BIT-POSITION>36</fx:BIT-POSITION>",
            ),
            "byte offset",
        ),
        (
            "pdu-order",
            text.replace(
                "<fx:PDU-REF ID-REF=\"P_Speed\"/>\n            <fx:BIT-POSITION>0</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>false</fx:IS-HIGH-LOW-BYTE-ORDER>",
                "<fx:PDU-REF ID-REF=\"P_Speed\"/>\n            <fx:BIT-POSITION>0</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>true</fx:IS-HIGH-LOW-BYTE-ORDER>",
            ),
            "high-low byte order PDU instances",
        ),
        (
            "identifier",
            text.replace(">419369524<", ">0x40000000<"),
            "not a CAN identifier",
        ),
    ];
    for (name, text, expected) in cases {
        assert_ne!(text, network(), "{name}: the fixture changed");
        let path = variant(name, &text);
        let e = load_err(&path, Some("Body"));
        assert!(e.contains(expected), "{name}: {e}");
        std::fs::remove_file(path).unwrap();
    }
}
