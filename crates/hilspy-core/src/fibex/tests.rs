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
            "nonlinear",
            text.replacen(
                "<ho:V>0</ho:V><ho:V>0.01</ho:V>",
                "<ho:V>0</ho:V><ho:V>0.01</ho:V><ho:V>1</ho:V>",
                1,
            ),
            "a rational function is RAT-FUNC",
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
            "does not fit in frame",
        ),
        (
            "split",
            text.replace(
                "<fx:PDU-REF ID-REF=\"P_Brake\"/>\n            <fx:BIT-POSITION>32</fx:BIT-POSITION>",
                "<fx:PDU-REF ID-REF=\"P_Brake\"/>\n            <fx:BIT-POSITION>28</fx:BIT-POSITION>",
            ),
            "BrakePressure of PDU BrakePdu: its bits are not one signal",
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

fn features() -> Database {
    load(data("fibex_features.xml"), None).unwrap()
}

fn named(v: &[(&str, Value)]) -> Vec<(String, Value)> {
    v.iter().map(|(n, v)| (n.to_string(), v.clone())).collect()
}

#[test]
fn sign_magnitude_and_ones_complement_codings() {
    let db = features();
    let m = db.message_by_name("Encodings").unwrap();
    let kinds: Vec<_> = m.signals.iter().map(|s| (s.kind, s.signed)).collect();
    assert_eq!(
        kinds,
        [
            (ValueKind::SignMagnitude, true),
            (ValueKind::OnesComplement, true),
            (ValueKind::SignMagnitude, true),
            (ValueKind::OnesComplement, true),
        ]
    );
    let values = named(&[
        ("SmSmall", Value::Int(-5)),
        ("OcSmall", Value::Int(-5)),
        ("SmWide", Value::Float(-100.5)),
        ("OcWide", Value::Int(-2)),
    ]);
    let data = m.encode(&values, &EncodeOptions::default()).unwrap();
    // -5: 0x85 (sign + magnitude), 0xFA (NOT 5); -100.5 / 0.5 = -201 =
    // 0x80C9 little endian; -2 = NOT 2 = 0xFFFD big endian in bytes 6-7.
    assert_eq!(data, [0x85, 0xFA, 0xC9, 0x80, 0, 0, 0xFF, 0xFD]);
    let decoded = m.decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded, values);
    // Negative zeros decode to 0.
    let zeros = m
        .decode(
            &[0x80, 0xFF, 0x00, 0x80, 0, 0, 0xFF, 0xFF],
            &DecodeOptions::default(),
        )
        .unwrap();
    assert!(
        zeros
            .iter()
            .all(|(_, v)| *v == Value::Int(0) || *v == Value::Float(0.0)),
        "{zeros:?}"
    );
}

#[test]
fn texttable_ranges_piecewise_and_rational_codings() {
    let db = features();
    let m = db.message_by_name("Codings").unwrap();
    // A single linear RAT-FUNC is a plain factor and offset.
    let ratio = m.signal("Ratio").unwrap();
    assert_eq!(
        (ratio.factor, ratio.offset, ratio.coding.is_none()),
        (1.5, 5.0, true)
    );
    let level = m.signal("Level").unwrap();
    assert!(level.coding.is_some());
    // The names, by their lower limits.
    assert_eq!(
        level.choices,
        [(0, "Low".into()), (4, "High".into()), (15, "Error".into())]
    );

    let decode = |data: [u8; 8]| m.decode(&data, &DecodeOptions::default()).unwrap();
    // Level 9 (High), Parabola 3: (3 - 8)^2, Curve 120: 2 * 120 - 150,
    // Inverse 9: 1000 / (9 + 1), Ratio 10: (10 + 3 * 10) / 2.
    assert_eq!(
        decode([0x39, 120, 9, 10, 0, 0, 0, 0]),
        named(&[
            ("Level", Value::Str("High".into())),
            ("Parabola", Value::Float(25.0)),
            ("Curve", Value::Float(90.0)),
            ("Inverse", Value::Float(100.0)),
            ("Ratio", Value::Float(20.0)),
        ])
    );
    // Level 3 (Low), Parabola 15 (its function's scale is open at 15),
    // Curve 99: 0.5 * 99, Inverse 0: 1000.
    assert_eq!(
        decode([0xF3, 99, 0, 0, 0, 0, 0, 0])[..4],
        named(&[
            ("Level", Value::Str("Low".into())),
            ("Parabola", Value::Str("NotAvailable".into())),
            ("Curve", Value::Float(49.5)),
            ("Inverse", Value::Float(1000.0)),
        ])
    );
    // Without names, a texttable-only raw value stays raw.
    let raw_names = m
        .decode(
            &[0xF3, 99, 0, 0, 0, 0, 0, 0],
            &DecodeOptions {
                decode_choices: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(raw_names[0].1, Value::Int(3));
    assert_eq!(raw_names[1].1, Value::Int(15));

    let values = named(&[
        ("Level", Value::Str("High".into())),
        ("Parabola", Value::Str("NotAvailable".into())),
        ("Curve", Value::Float(90.0)),
        ("Inverse", Value::Float(100.0)),
        ("Ratio", Value::Float(20.0)),
    ]);
    let opts = EncodeOptions::default();
    // Names encode to their lower limits; numbers invert within a scale.
    assert_eq!(
        m.encode(&values, &opts).unwrap(),
        [0xF4, 120, 9, 10, 0, 0, 0, 0]
    );
    for (curve, raw) in [(0.0, 0u8), (49.5, 99), (52.0, 101), (360.0, 255)] {
        let mut v = values.clone();
        v[2].1 = Value::Float(curve);
        assert_eq!(m.encode(&v, &opts).unwrap()[1], raw, "{curve}");
    }
    for (inverse, raw) in [(1000.0, 0u8), (500.0, 1), (101.0, 9), (1000.0 / 256.0, 255)] {
        let mut v = values.clone();
        v[3].1 = Value::Float(inverse);
        assert_eq!(m.encode(&v, &opts).unwrap()[2], raw, "{inverse}");
    }
    let err = |i: usize, value: Value| {
        let mut v = values.clone();
        v[i].1 = value;
        m.encode(&v, &opts).unwrap_err().to_string()
    };
    // Physical constraints still apply; a value no scale reaches fails.
    assert!(err(2, Value::Float(361.0)).contains("maximum"));
    let e = err(3, Value::Float(2000.0));
    assert!(e.contains("signal Inverse") && e.contains("outside"), "{e}");
    // (x - 8)^2 is not monotonic over its scale: it decodes but cannot encode.
    let e = err(1, Value::Float(25.0));
    assert!(
        e.contains("signal Parabola") && e.contains("monotonic"),
        "{e}"
    );
    assert!(err(0, Value::Str("Medium".into())).contains("Low, High, Error"));
}

/// The frames of the features fixture that a DBC can express match it.
#[test]
fn features_match_the_equivalent_dbc() {
    let fibex = features();
    let dbc = Database::from_file(data("fibex_features.dbc")).unwrap();
    for d in &dbc.messages {
        let f = fibex.message_by_name(&d.name).unwrap();
        assert_eq!((f.id, f.size), (d.id, d.size), "{}", d.name);
        let mut expected = d.signals.clone();
        for s in &mut expected {
            s.receivers.clear();
        }
        assert_eq!(f.signals, expected, "{}", d.name);
    }
}

#[test]
fn pdus_at_bit_offsets() {
    let db = features();
    let m = db.message_by_name("Shifted").unwrap();
    let layout: Vec<_> = m
        .signals
        .iter()
        .map(|s| (s.name.as_str(), s.start_bit, s.byte_order))
        .collect();
    assert_eq!(
        layout,
        [
            ("ShiftedA", 4, ByteOrder::LittleEndian),
            ("ShiftedB", 19, ByteOrder::BigEndian),
            ("Tail", 27, ByteOrder::LittleEndian),
        ]
    );
    let values = named(&[
        ("ShiftedA", Value::Int(0x3A5)),
        ("ShiftedB", Value::Int(0xC)),
        ("Tail", Value::Int(0xB7)),
    ]);
    let data = m.encode(&values, &EncodeOptions::default()).unwrap();
    // A: bits 4..14; B: byte 2 bits 3..0; Tail: bits 27..35.
    assert_eq!(data, [0x50, 0x3A, 0x0C, 0xB8, 0x05, 0, 0, 0]);
    assert_eq!(m.decode(&data, &DecodeOptions::default()).unwrap(), values);
}
