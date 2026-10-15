use super::*;

fn db() -> Database {
    Database::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/data/vehicle.dbc"
    ))
    .unwrap()
}

fn values(v: &[(&str, Value)]) -> Vec<(String, Value)> {
    v.iter().map(|(n, v)| (n.to_string(), v.clone())).collect()
}

#[test]
fn loads_attributes_and_comments() {
    let db = db();
    assert_eq!(db.messages.len(), 5);
    assert_eq!(db.nodes, vec!["VCU", "ADAS", "GNSS"]);
    let m = db.message_by_name("VehicleSpeed").unwrap();
    assert_eq!((m.id, m.extended, m.size), (0x100, false, 8));
    assert_eq!(m.cycle_time, Some(std::time::Duration::from_millis(10)));
    assert_eq!(m.comment.as_deref(), Some("Vehicle speed from the VCU"));
    let gear = m.signal("Gear").unwrap();
    assert_eq!(gear.choices[3], (3, "D".to_string()));
    assert_eq!(gear.initial_raw, Some(3.0));
    let diag = db.message_by_id(0x8010_0000 & 0x1FFF_FFFF, true).unwrap();
    assert_eq!(diag.name, "Diagnostic");
    assert!(db.message_by_name("RadarObjects").unwrap().fd);
    assert!(!m.fd);
}

#[test]
fn transmitters_and_receivers() {
    let db = db();
    let m = db.message_by_name("VehicleSpeed").unwrap();
    assert_eq!(
        (m.transmitters.as_slice(), m.receivers.as_slice()),
        (&["VCU".to_string()][..], &["ADAS".to_string()][..])
    );
    assert_eq!(db.ecus(), ["VCU", "ADAS", "GNSS"]);
    // BO_TX_BU_ adds transmitters; Vector__XXX is none; receivers are
    // those of every signal, once each; ECUs not in BU_ still count.
    let text = "VERSION \"\"\n\nNS_ :\n\nBS_:\n\nBU_: A B\n\n\
        BO_ 1 M: 8 A\n SG_ S : 0|8@1+ (1,0) [0|0] \"\" B,C\n SG_ T : 8|8@1+ (1,0) [0|0] \"\" C\n\n\
        BO_ 2 N: 8 Vector__XXX\n SG_ U : 0|8@1+ (1,0) [0|0] \"\" Vector__XXX\n\n\
        BO_TX_BU_ 1 : D,A;\n";
    let db = Database::parse(text).unwrap();
    let m = db.message_by_name("M").unwrap();
    assert_eq!(m.transmitters, ["A", "D"]);
    assert_eq!(m.receivers, ["B", "C"]);
    let n = db.message_by_name("N").unwrap();
    assert!(n.transmitters.is_empty() && n.receivers.is_empty());
    assert_eq!(db.ecus(), ["A", "B", "D", "C"]);
}

#[test]
fn little_endian_round_trip() {
    let db = db();
    let m = db.message_by_name("VehicleSpeed").unwrap();
    let data = m
        .encode(
            &values(&[
                ("Speed", Value::Float(123.45)),
                ("Accel", Value::Float(-1.5)),
                ("Gear", Value::Str("D".into())),
                ("Counter", Value::Int(9)),
            ]),
            &EncodeOptions::default(),
        )
        .unwrap();
    // Speed raw 12345 = 0x3039; Accel raw (-1.5 + 2) / 0.001 = 500 = 0x01F4.
    assert_eq!(data, vec![0x39, 0x30, 0xF4, 0x01, 0x03, 0x00, 0x00, 0x90]);
    let decoded = m.decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[0], ("Speed".into(), Value::Float(123.45)));
    assert!(matches!(decoded[1].1, Value::Float(a) if (a + 1.5).abs() < 1e-9));
    assert_eq!(decoded[2], ("Gear".into(), Value::Str("D".into())));
    assert_eq!(decoded[3], ("Counter".into(), Value::Int(9)));
}

#[test]
fn big_endian_round_trip() {
    let db = db();
    let m = db.message_by_name("SteeringCmd").unwrap();
    let v = values(&[
        ("SteeringAngle", Value::Float(-90.5)),
        ("Torque", Value::Float(1.25)),
        ("Enable", Value::Int(1)),
        ("Checksum", Value::Int(0xA5)),
    ]);
    let data = m.encode(&v, &EncodeOptions::default()).unwrap();
    // Angle raw -905 = 0xFC77 in bytes 0-1; Torque raw 2125 = 0x84D in byte 2
    // and the high nibble of byte 3, Enable at bit 27, Checksum in byte 7.
    assert_eq!(data, vec![0xFC, 0x77, 0x84, 0xD8, 0, 0, 0, 0xA5]);
    let decoded = m.decode(&data, &DecodeOptions::default()).unwrap();
    assert!(matches!(decoded[0].1, Value::Float(a) if (a + 90.5).abs() < 1e-9));
    assert!(matches!(decoded[1].1, Value::Float(t) if (t - 1.25).abs() < 1e-9));
    assert_eq!(decoded[2].1, Value::Int(1));
    assert_eq!(decoded[3].1, Value::Int(0xA5));
}

#[test]
fn strict_checks() {
    let db = db();
    let m = db.message_by_name("VehicleSpeed").unwrap();
    let opts = EncodeOptions::default();
    // Missing signals.
    assert!(m
        .encode(&values(&[("Speed", Value::Float(1.0))]), &opts)
        .is_err());
    // Out of [min|max].
    let mut v = values(&[
        ("Speed", Value::Float(700.0)),
        ("Accel", Value::Float(0.0)),
        ("Gear", Value::Int(0)),
        ("Counter", Value::Int(0)),
    ]);
    assert!(m.encode(&v, &opts).is_err());
    // Unknown choice and unknown signal.
    v[0].1 = Value::Float(1.0);
    v[2].1 = Value::Str("X".into());
    assert!(m.encode(&v, &opts).is_err());
    assert!(m
        .encode(&values(&[("Nope", Value::Int(0))]), &opts)
        .is_err());
    // Lenient: missing signals take GenSigStartValue (Gear = 3) or 0.
    let lenient = EncodeOptions {
        strict: false,
        ..opts
    };
    let data = m
        .encode(&values(&[("Speed", Value::Float(1.0))]), &lenient)
        .unwrap();
    assert_eq!(data, vec![0x64, 0, 0, 0, 0x03, 0, 0, 0]);
    // Raw overflow fails even when lenient.
    assert!(m
        .encode(&values(&[("Counter", Value::Int(16))]), &lenient)
        .is_err());
}

#[test]
fn multiplexing() {
    let db = db();
    let m = db.message_by_name("Diagnostic").unwrap();
    let opts = EncodeOptions::default();
    let data = m
        .encode(
            &values(&[
                ("Mode", Value::Str("Power".into())),
                ("Voltage", Value::Float(12.5)),
                ("ErrorCode", Value::Int(7)),
            ]),
            &opts,
        )
        .unwrap();
    assert_eq!(&data[..4], &[2, 0xD4, 0x30, 7]);
    let decoded = m.decode(&data, &DecodeOptions::default()).unwrap();
    let names: Vec<_> = decoded.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["Mode", "Voltage", "ErrorCode"]);
    // Temperature is not present for Mode = 2.
    assert!(m
        .encode(
            &values(&[("Mode", Value::Int(2)), ("Temperature", Value::Float(1.0))]),
            &opts
        )
        .is_err());
    let data = m
        .encode(
            &values(&[
                ("Mode", Value::Int(1)),
                ("Temperature", Value::Float(-12.3)),
            ]),
            &opts,
        )
        .unwrap();
    let decoded = m.decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[0].1, Value::Str("Thermal".into()));
    assert_eq!(decoded.len(), 2);
}

#[test]
fn floats_and_fd() {
    let db = db();
    let imu = db.message_by_name("ImuData").unwrap();
    let data = imu
        .encode(
            &values(&[
                ("YawRate", Value::Float(0.25)),
                ("LatAccel", Value::Float(-9.5)),
            ]),
            &EncodeOptions::default(),
        )
        .unwrap();
    assert_eq!(&data[..4], &0.25f32.to_le_bytes());
    let decoded = imu.decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[1].1, Value::Float(-9.5));

    let radar = db.message_by_name("RadarObjects").unwrap();
    let data = radar
        .encode(
            &values(&[
                ("ObjectCount", Value::Int(3)),
                ("Distance", Value::Float(42.0)),
                ("Tail", Value::Int(0xEE)),
            ]),
            &EncodeOptions::default(),
        )
        .unwrap();
    assert!(radar.fd);
    assert_eq!(data.len(), 64);
    assert_eq!(data[63], 0xEE);
    let decoded = radar.decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[2].1, Value::Int(0xEE));
    // A short payload fails unless truncation is allowed.
    assert!(radar.decode(&data[..8], &DecodeOptions::default()).is_err());
    let partial = radar
        .decode(
            &data[..8],
            &DecodeOptions {
                allow_truncated: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(partial.len(), 2);
}

#[test]
fn raw_mode() {
    let db = db();
    let m = db.message_by_name("VehicleSpeed").unwrap();
    let data = m
        .encode(
            &values(&[
                ("Speed", Value::Int(12345)),
                ("Accel", Value::Int(500)),
                ("Gear", Value::Int(3)),
                ("Counter", Value::Int(9)),
            ]),
            &EncodeOptions {
                scaling: false,
                strict: false,
            },
        )
        .unwrap();
    let decoded = m
        .decode(
            &data,
            &DecodeOptions {
                decode_choices: false,
                scaling: false,
                allow_truncated: false,
            },
        )
        .unwrap();
    assert_eq!(decoded[0].1, Value::Int(12345));
    assert_eq!(decoded[1].1, Value::Int(500));
    assert_eq!(decoded[2].1, Value::Int(3));
}

#[test]
fn parse_errors() {
    assert!(Database::parse("this is not a dbc").is_err());
    let out_of_bounds = "VERSION \"\"\n\nNS_ :\n\nBS_:\n\nBU_:\n\nBO_ 1 M: 1 Vector__XXX\n SG_ S : 4|8@1+ (1,0) [0|0] \"\" Vector__XXX\n";
    assert!(Database::parse(out_of_bounds).is_err());
}

#[test]
fn raw_values_are_range_checked_physically() {
    let db = db();
    let m = db.message_by_name("VehicleSpeed").unwrap();
    let raw = EncodeOptions {
        scaling: false,
        strict: true,
    };
    let v = |speed: i64| {
        values(&[
            ("Speed", Value::Int(speed)),
            ("Accel", Value::Int(2000)),
            ("Gear", Value::Int(0)),
            ("Counter", Value::Int(0)),
        ])
    };
    assert!(m.encode(&v(65535), &raw).is_ok());
    // Accel raw 2000 is 0.0 m/s^2; raw 33000 would be 31 (> 30).
    let mut over = v(0);
    over[1].1 = Value::Int(33000);
    assert!(m.encode(&over, &raw).is_err());
}

/// A one-signal message: an integer at bit 0, little endian.
fn single(kind: ValueKind, size: u32) -> Message {
    let mut signal = convert_single(size);
    signal.kind = kind;
    signal.signed = true;
    Message {
        id: 1,
        extended: false,
        name: "M".into(),
        size: 8,
        transmitters: Vec::new(),
        receivers: Vec::new(),
        signals: vec![signal],
        comment: None,
        cycle_time: None,
        fd: false,
    }
}

fn convert_single(size: u32) -> Signal {
    let text = format!(
        "VERSION \"\"\n\nNS_ :\n\nBS_:\n\nBU_:\n\nBO_ 1 M: 8 Vector__XXX\n SG_ S : 0|{size}@1- (1,0) [0|0] \"\" Vector__XXX\n"
    );
    Database::parse(&text).unwrap().messages[0].signals[0].clone()
}

#[test]
fn sign_magnitude_and_ones_complement() {
    let raw = |m: &Message, v: i64| {
        let data = m
            .encode(&values(&[("S", Value::Int(v))]), &EncodeOptions::default())
            .map(|d| u64::from_le_bytes(d.try_into().unwrap()));
        data.map_err(|e| e.to_string())
    };
    let decode = |m: &Message, bits: u64| {
        m.decode(&bits.to_le_bytes(), &DecodeOptions::default())
            .unwrap()[0]
            .1
            .clone()
    };
    let sm = single(ValueKind::SignMagnitude, 8);
    assert_eq!(raw(&sm, 0), Ok(0x00));
    assert_eq!(raw(&sm, 127), Ok(0x7F));
    assert_eq!(raw(&sm, -127), Ok(0xFF));
    assert_eq!(raw(&sm, -1), Ok(0x81));
    assert!(raw(&sm, 128).unwrap_err().contains("sign-magnitude"));
    assert!(raw(&sm, -128).is_err());
    assert_eq!(decode(&sm, 0xFF), Value::Int(-127));
    assert_eq!(decode(&sm, 0x81), Value::Int(-1));
    assert_eq!(decode(&sm, 0x7F), Value::Int(127));
    // Negative zero.
    assert_eq!(decode(&sm, 0x80), Value::Int(0));

    let oc = single(ValueKind::OnesComplement, 8);
    assert_eq!(raw(&oc, 0), Ok(0x00));
    assert_eq!(raw(&oc, 127), Ok(0x7F));
    assert_eq!(raw(&oc, -127), Ok(0x80));
    assert_eq!(raw(&oc, -1), Ok(0xFE));
    assert!(raw(&oc, -128).unwrap_err().contains("one's complement"));
    assert_eq!(decode(&oc, 0x80), Value::Int(-127));
    assert_eq!(decode(&oc, 0xFE), Value::Int(-1));
    assert_eq!(decode(&oc, 0xFF), Value::Int(0));

    // 64 bits: the magnitude is 63 bits wide.
    let max = i64::MAX;
    let sm = single(ValueKind::SignMagnitude, 64);
    assert_eq!(raw(&sm, -max), Ok(u64::MAX));
    assert_eq!(decode(&sm, u64::MAX), Value::Int(-max));
    assert_eq!(decode(&sm, 1 << 63), Value::Int(0));
    let oc = single(ValueKind::OnesComplement, 64);
    assert_eq!(raw(&oc, -max), Ok(1 << 63));
    assert_eq!(decode(&oc, 1 << 63), Value::Int(-max));
    assert_eq!(decode(&oc, u64::MAX), Value::Int(0));
    assert_eq!(raw(&oc, max), Ok(max as u64));
}
