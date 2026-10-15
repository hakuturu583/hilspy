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
    assert_eq!(fibex.nodes, dbc.nodes);
    for d in &dbc.messages {
        let f = fibex.message_by_name(&d.name).unwrap();
        assert_eq!(
            (f.id, f.extended, f.size, f.fd, &f.comment, f.cycle_time),
            (d.id, d.extended, d.size, d.fd, &d.comment, d.cycle_time),
            "{}",
            d.name
        );
        assert_eq!(
            (&f.transmitters, &f.receivers),
            (&d.transmitters, &d.receivers),
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
fn cycle_times_from_cyclic_timings() {
    let ms = |n| Some(std::time::Duration::from_millis(n));
    let body = load(data("network.xml"), Some("Body")).unwrap();
    let cycle = |db: &Database, name: &str| db.message_by_name(name).unwrap().cycle_time;
    assert_eq!(cycle(&body, "VehicleStatus"), ms(10));
    assert_eq!(cycle(&body, "SteeringCmd"), None);
    assert_eq!(cycle(&body, "RadarObjects"), ms(50));
    // Timing belongs to the triggering: the same frame on the chassis
    // channel has none.
    let chassis = load(data("network.xml"), Some("Chassis")).unwrap();
    assert_eq!(cycle(&chassis, "VehicleStatus"), None);
    // A zero time is no cycle time; a bad one is an error naming it.
    let with = |value: &str| {
        let text = network().replace(
            "<fx:VALUE>PT0.01S</fx:VALUE>",
            &format!("<fx:VALUE>{value}</fx:VALUE>"),
        );
        variant(&format!("cycle-{}", value.len()), &text)
    };
    let path = with("PT0S");
    assert_eq!(
        cycle(&load(&path, Some("Body")).unwrap(), "VehicleStatus"),
        None
    );
    std::fs::remove_file(path).unwrap();
    let path = with("P1M");
    let e = load_err(&path, Some("Body"));
    assert!(
        e.contains("FT_VehicleStatus_Body") && e.contains("\"P1M\""),
        "{e}"
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn ecus_from_connector_ports() {
    let names = |v: &[String]| v.join(",");
    let chassis = load(data("network.xml"), Some("Chassis")).unwrap();
    assert_eq!(chassis.nodes, ["Gateway", "Adas", "Radar", "Abs"]);
    // Ports refer to frame triggerings, so the same frame has other ECUs on
    // the other channel.
    let status = chassis.message_by_name("VehicleStatus").unwrap();
    assert_eq!(
        (names(&status.transmitters), names(&status.receivers)),
        ("Gateway".into(), "Abs".into())
    );
    let wheels = chassis.message_by_name("WheelSpeeds").unwrap();
    assert_eq!(
        (names(&wheels.transmitters), names(&wheels.receivers)),
        ("Abs".into(), "Gateway".into())
    );
    let body = load(data("network.xml"), Some("Body")).unwrap();
    let radar = body.message_by_name("RadarObjects").unwrap();
    assert_eq!(
        (names(&radar.transmitters), names(&radar.receivers)),
        ("Radar".into(), "Adas".into())
    );
    // Several senders; an ECU without a short name is named by its ID;
    // without ECUs a frame has neither.
    let text = network()
        .replace(
            "<fx:CONNECTOR ID=\"CON_Radar_Body\" xsi:type=\"can:CONNECTOR-TYPE\">\n            <fx:CHANNEL-REF ID-REF=\"CH_BODY\"/>",
            "<fx:CONNECTOR ID=\"CON_Radar_Body\" xsi:type=\"can:CONNECTOR-TYPE\">\n            <fx:CHANNEL-REF ID-REF=\"CH_BODY\"/>\n            <fx:INPUTS><fx:INPUT-PORT ID=\"IP_X\"><fx:FRAME-TRIGGERING-REF ID-REF=\"FT_SteeringCmd\"/></fx:INPUT-PORT></fx:INPUTS>",
        )
        .replace(
            "<ho:SHORT-NAME>Abs</ho:SHORT-NAME>",
            "",
        )
        .replace(
            "<fx:OUTPUT-PORT ID=\"OP_Radar_Body_RadarObjects\">",
            "<fx:OUTPUT-PORT ID=\"OP_Gw_Radar\"><fx:FRAME-TRIGGERING-REF ID-REF=\"FT_VehicleStatus_Body\"/></fx:OUTPUT-PORT><fx:OUTPUT-PORT ID=\"OP_Radar_Body_RadarObjects\">",
        );
    let path = variant("ecus", &text);
    let db = load(&path, Some("Body")).unwrap();
    assert_eq!(db.nodes, ["Gateway", "Adas", "Radar", "ECU_ABS"]);
    let status = db.message_by_name("VehicleStatus").unwrap();
    assert_eq!(names(&status.transmitters), "Gateway,Radar");
    let steer = db.message_by_name("SteeringCmd").unwrap();
    assert_eq!(names(&steer.receivers), "Gateway,Radar");
    std::fs::remove_file(path).unwrap();
    let text = without(&network(), "<fx:ECUS>", "</fx:ECUS>");
    let path = variant("no-ecus", &text);
    let db = load(&path, Some("Body")).unwrap();
    assert!(db.nodes.is_empty());
    assert!(db
        .messages
        .iter()
        .all(|m| m.transmitters.is_empty() && m.receivers.is_empty()));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn xs_durations() {
    use std::time::Duration;
    let ok = |s: &str| xs_duration(s).unwrap_or_else(|| panic!("{s}"));
    assert_eq!(ok("PT0.01S"), Duration::from_millis(10));
    assert_eq!(ok("PT0.0005S"), Duration::from_micros(500));
    assert_eq!(ok("PT1.000000001S"), Duration::from_nanos(1_000_000_001));
    assert_eq!(ok("PT10S"), Duration::from_secs(10));
    assert_eq!(ok("PT0S"), Duration::ZERO);
    assert_eq!(ok("PT1M30S"), Duration::from_secs(90));
    assert_eq!(ok("PT2H"), Duration::from_secs(7200));
    assert_eq!(ok("P1DT1S"), Duration::from_secs(86_401));
    assert_eq!(ok("P1D"), Duration::from_secs(86_400));
    for bad in [
        "", "P", "PT", "0.01", "PT0.01", "-PT1S", "P1Y", "P1M", "PT1S1M", "PT1.5M", "PT.5S",
        "PT1..5S", "PTS", "P1DT", "PT1S2S", "pt1s",
    ] {
        assert_eq!(xs_duration(bad), None, "{bad}");
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

#[test]
fn high_low_pdu_instances() {
    let db = features();
    let m = db.message_by_name("Swapped").unwrap();
    let layout: Vec<_> = m
        .signals
        .iter()
        .map(|s| (s.name.as_str(), s.start_bit, s.byte_order))
        .collect();
    // SwappedPdu (4 bytes, MSB at bit 15) fills bytes 1-4, its last byte
    // first: Intel SwapA (PDU bits 0-11) becomes Motorola from frame bit 27,
    // Motorola SwapB (PDU bytes 2-3) becomes Intel from frame bit 8.
    // NarrowPdu (1 byte, MSB at bit 45) is bits 45..40 and 55..54.
    assert_eq!(
        layout,
        [
            ("SwapA", 27, ByteOrder::BigEndian),
            ("SwapB", 8, ByteOrder::LittleEndian),
            ("SwapC", 45, ByteOrder::BigEndian),
        ]
    );
    let values = named(&[
        ("SwapA", Value::Int(0xABC)),
        ("SwapB", Value::Int(0x1234)),
        ("SwapC", Value::Int(0x5A)),
    ]);
    let payload = m.encode(&values, &EncodeOptions::default()).unwrap();
    // The PDU's bytes BC 0A 12 34 (A little endian, B big endian), reversed
    // into frame bytes 1-4; 0x5A = 010110 in byte 5 bits 5..0, 10 in byte 6
    // bits 7..6.
    assert_eq!(payload, [0x00, 0x34, 0x12, 0x0A, 0xBC, 0x16, 0x80, 0x00]);
    assert_eq!(
        m.decode(&payload, &DecodeOptions::default()).unwrap(),
        values
    );

    // Low-high instances are unchanged: the same PDU low-high at byte 1.
    let text = std::fs::read_to_string(data("fibex_features.xml"))
        .unwrap()
        .replace(
            "<fx:BIT-POSITION>15</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>true",
            "<fx:BIT-POSITION>8</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>false",
        );
    let path = variant("low-high", &text);
    let db = load(&path, None).unwrap();
    std::fs::remove_file(path).unwrap();
    let m = db.message_by_name("Swapped").unwrap();
    let payload = m.encode(&values, &EncodeOptions::default()).unwrap();
    assert_eq!(&payload[..5], [0x00, 0xBC, 0x0A, 0x12, 0x34]);

    // Off a byte boundary, the Motorola signal across PDU bytes is split.
    let text = std::fs::read_to_string(data("fibex_features.xml"))
        .unwrap()
        .replace(
            "<fx:BIT-POSITION>15</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>true",
            "<fx:BIT-POSITION>13</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>true",
        );
    let path = variant("high-low-split", &text);
    let e = load_err(&path, None);
    std::fs::remove_file(path).unwrap();
    assert!(
        e.contains("signal SwapB of PDU SwappedPdu: its bits are not one signal"),
        "{e}"
    );
}

#[test]
fn multiplexed_pdus() {
    let db = features();
    let m = db.message_by_name("Multiplexed").unwrap();
    let mux: Vec<_> = m
        .signals
        .iter()
        .map(|s| {
            (
                s.name.as_str(),
                s.is_multiplexor,
                s.multiplexing
                    .as_ref()
                    .map(|x| (x.multiplexor, x.ranges.clone())),
            )
        })
        .collect();
    assert_eq!(
        mux,
        [
            ("Mux", true, None),
            ("A16", false, Some((0, vec![(1, 1)]))),
            ("A8", false, Some((0, vec![(1, 1)]))),
            ("B16", false, Some((0, vec![(2, 3)]))),
            ("B4", false, Some((0, vec![(2, 3)]))),
            ("Counter", false, None),
        ]
    );
    let opts = EncodeOptions::default();
    let cases = [
        (
            named(&[
                ("Mux", Value::Int(1)),
                ("A16", Value::Int(0x1234)),
                ("A8", Value::Int(0xAB)),
                ("Counter", Value::Int(7)),
            ]),
            [1, 0x34, 0x12, 0xAB, 0, 0, 7, 0],
        ),
        (
            named(&[
                ("Mux", Value::Int(2)),
                ("B16", Value::Int(0xBEEF)),
                ("B4", Value::Int(0x9)),
                ("Counter", Value::Int(8)),
            ]),
            [2, 0xEF, 0xBE, 0x09, 0, 0, 8, 0],
        ),
        (
            named(&[
                ("Mux", Value::Int(3)),
                ("B16", Value::Int(1)),
                ("B4", Value::Int(0xF)),
                ("Counter", Value::Int(9)),
            ]),
            [3, 0x01, 0x00, 0x0F, 0, 0, 9, 0],
        ),
    ];
    for (values, bytes) in cases {
        let data = m.encode(&values, &opts).unwrap();
        assert_eq!(data, bytes, "{values:?}");
        assert_eq!(m.decode(&data, &DecodeOptions::default()).unwrap(), values);
    }
    // A switched PDU's signal is absent under another code.
    let e = m
        .encode(
            &named(&[
                ("Mux", Value::Int(1)),
                ("A16", Value::Int(0)),
                ("A8", Value::Int(0)),
                ("B4", Value::Int(0)),
                ("Counter", Value::Int(0)),
            ]),
            &opts,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("B4") && e.contains("not present"), "{e}");
    // Only the switch and the static part under an unknown code.
    let decoded = m
        .decode(&[9, 1, 2, 3, 0, 0, 5, 0], &DecodeOptions::default())
        .unwrap();
    assert_eq!(
        decoded,
        named(&[("Mux", Value::Int(9)), ("Counter", Value::Int(5))])
    );
}

#[test]
fn unsupported_multiplexing_is_reported() {
    let text = std::fs::read_to_string(data("fibex_features.xml")).unwrap();
    let cases = [
        (
            "code",
            text.replace(
                "<fx:SWITCH-CODE>3</fx:SWITCH-CODE>",
                "<fx:SWITCH-CODE>256</fx:SWITCH-CODE>",
            ),
            "switch code 256 does not fit in the 8 bits of switch Mux",
        ),
        (
            "duplicate",
            text.replace(
                "<fx:SWITCH-CODE>3</fx:SWITCH-CODE>",
                "<fx:SWITCH-CODE>2</fx:SWITCH-CODE>",
            ),
            "distinct integer SWITCH-CODE",
        ),
        (
            "position",
            text.replace(
                "<fx:PDU-REF ID-REF=\"P_MuxA\"/>",
                "<fx:PDU-REF ID-REF=\"P_MuxA\"/><fx:BIT-POSITION>8</fx:BIT-POSITION>",
            ),
            "must be at bit 0",
        ),
        (
            "nested",
            text.replace(
                "<fx:PDU-REF ID-REF=\"P_MuxStatic\"/>",
                "<fx:PDU-REF ID-REF=\"P_Mux\"/>",
            ),
            "nested multiplexing",
        ),
        (
            "switch",
            text.replace(
                "<fx:BIT-POSITION>0</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>false</fx:IS-HIGH-LOW-BYTE-ORDER>\n            <ho:BIT-LENGTH>8</ho:BIT-LENGTH>",
                "<fx:BIT-POSITION>60</fx:BIT-POSITION>\n            <fx:IS-HIGH-LOW-BYTE-ORDER>false</fx:IS-HIGH-LOW-BYTE-ORDER>\n            <ho:BIT-LENGTH>8</ho:BIT-LENGTH>",
            ),
            "switch Mux from bit 60 do not fit",
        ),
        (
            "dynamic",
            without(&text, "<fx:DYNAMIC-PART ID=\"DP_Mux\">", "</fx:DYNAMIC-PART>"),
            "no DYNAMIC-PART",
        ),
    ];
    for (name, changed, expected) in cases {
        assert_ne!(changed, text, "{name}: the fixture changed");
        let path = variant(name, &changed);
        let e = load_err(&path, None);
        std::fs::remove_file(path).unwrap();
        assert!(e.contains(expected), "{name}: {e}");
    }
}

fn lin_flexray() -> String {
    std::fs::read_to_string(data("lin_flexray.xml")).unwrap()
}

fn lin(channel: &ChannelDescription) -> &Lin {
    match &channel.bus {
        Bus::Lin(l) => l,
        other => panic!("not LIN: {other:?}"),
    }
}

#[test]
fn lin_frames_and_checksum_models() {
    let ch = load_channel(data("lin_flexray.xml"), Some("Lin"), None).unwrap();
    let l = lin(&ch);
    let frames: Vec<_> = l
        .frames
        .iter()
        .map(|f| (f.name.as_str(), f.id, f.checksum))
        .collect();
    assert_eq!(
        frames,
        [
            ("LightCmd", 0x10, LinChecksum::Enhanced), // LIN 2.1 default
            ("LightStatus", 0x11, LinChecksum::Classic), // CHECKSUM-TYPE
            ("MasterReq", 0x3C, LinChecksum::Classic), // diagnostic
        ]
    );
    let cmd = ch.database.message_by_id(0x10, false).unwrap();
    assert_eq!(
        (cmd.name.as_str(), cmd.size, cmd.fd),
        ("LightCmd", 2, false)
    );
    assert_eq!(cmd.transmitters, ["Bcm"]);
    assert_eq!(cmd.receivers, ["Lamp"]);
    assert_eq!(cmd.signal("Brightness").unwrap().factor, 0.5);
    // The CAN-only entry point gives the same database.
    let db = load(data("lin_flexray.xml"), Some("CH_LIN")).unwrap();
    assert_eq!(db.messages, ch.database.messages);
    // LIN 1.x clusters default to classic.
    let path = variant(
        "lin13",
        &lin_flexray().replace(
            "<fx:PROTOCOL-VERSION>2.1</fx:PROTOCOL-VERSION>\n        <fx:CHANNEL-REFS>\n          <fx:CHANNEL-REF ID-REF=\"CH_LIN\"/>",
            "<fx:PROTOCOL-VERSION>1.3</fx:PROTOCOL-VERSION>\n        <fx:CHANNEL-REFS>\n          <fx:CHANNEL-REF ID-REF=\"CH_LIN\"/>",
        ),
    );
    let ch = load_channel(&path, Some("Lin"), None).unwrap();
    assert_eq!(lin(&ch).frames[0].checksum, LinChecksum::Classic);
}

#[test]
fn lin_schedule_tables_give_cycle_times() {
    let ms = Duration::from_millis;
    let ch = load_channel(data("lin_flexray.xml"), Some("Lin"), None).unwrap();
    let tables = &lin(&ch).schedule_tables;
    assert_eq!(tables.len(), 2);
    let (normal, diag) = (&tables[0], &tables[1]);
    assert_eq!((normal.name.as_str(), normal.period()), ("Normal", ms(40)));
    assert_eq!(normal.entries[1].frame.as_deref(), Some("LightStatus"));
    assert_eq!((diag.name.as_str(), diag.period()), ("Diagnostic", ms(20)));
    assert_eq!(diag.entries[1].frame, None, "a free-format entry");
    // LightCmd is twice in Normal's 40 ms; each frame from the first table
    // that sends it.
    let cycle =
        |ch: &ChannelDescription, name: &str| ch.database.message_by_name(name).unwrap().cycle_time;
    assert_eq!(cycle(&ch, "LightCmd"), Some(ms(20)));
    assert_eq!(cycle(&ch, "LightStatus"), Some(ms(40)));
    assert_eq!(cycle(&ch, "MasterReq"), Some(ms(20)));
    // A chosen table (by name or ID) gives the cycle times alone.
    let d = load_channel(data("lin_flexray.xml"), Some("Lin"), Some("Diagnostic")).unwrap();
    assert_eq!(cycle(&d, "MasterReq"), Some(ms(20)));
    assert_eq!(cycle(&d, "LightCmd"), None);
    let n = load_channel(data("lin_flexray.xml"), Some("Lin"), Some("ST_Normal")).unwrap();
    assert_eq!(cycle(&n, "MasterReq"), None);
    assert_eq!(cycle(&n, "LightCmd"), Some(ms(20)));
    let e = load_channel(data("lin_flexray.xml"), Some("Lin"), Some("Sleep"))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("no schedule table \"Sleep\"") && e.contains("tables: Normal, Diagnostic"),
        "{e}"
    );
    let e = load_channel(data("network.xml"), Some("Body"), Some("Normal"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a LIN channel"), "{e}");
}

#[test]
fn lin_errors_are_reported() {
    let text = lin_flexray();
    let cases = [
        (
            text.replace(
                "<fx:IDENTIFIER-VALUE>16</fx:IDENTIFIER-VALUE>",
                "<fx:IDENTIFIER-VALUE>64</fx:IDENTIFIER-VALUE>",
            ),
            "not a LIN frame identifier (0..=63)",
        ),
        (
            text.replace(
                "<lin:CHECKSUM-TYPE>CLASSIC</lin:CHECKSUM-TYPE>",
                "<lin:CHECKSUM-TYPE>FANCY</lin:CHECKSUM-TYPE>",
            ),
            "\"FANCY\" is not CLASSIC or ENHANCED",
        ),
        (
            text.replacen(
                "<lin:DELAY>PT0.01S</lin:DELAY>",
                "<lin:DELAY>10</lin:DELAY>",
                1,
            ),
            "schedule table Normal: delay \"10\" is not a duration",
        ),
        (
            text.replace(
                "<fx:FRAME-TRIGGERING-REF ID-REF=\"FT_LightStatus\"/>\n              </lin:SCHEDULE",
                "<fx:FRAME-TRIGGERING-REF ID-REF=\"FT_Nope\"/>\n              </lin:SCHEDULE",
            ),
            "refers to frame triggering FT_Nope",
        ),
        (
            text.replace(
                "<fx:BYTE-LENGTH>8</fx:BYTE-LENGTH>\n        <fx:FRAME-TYPE>",
                "<fx:BYTE-LENGTH>9</fx:BYTE-LENGTH>\n        <fx:FRAME-TYPE>",
            ),
            "LIN frame MasterReq is 9 bytes (at most 8)",
        ),
    ];
    for (i, (text, expected)) in cases.into_iter().enumerate() {
        let path = variant(&format!("lin-error-{i}"), &text);
        let e = load_channel(&path, Some("Lin"), None)
            .unwrap_err()
            .to_string();
        assert!(e.contains(expected), "{expected}: {e}");
    }
}
