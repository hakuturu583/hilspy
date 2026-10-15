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
    assert_eq!(m.cycle_time_ms, Some(10));
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
    let frame = radar
        .encode_frame(
            &values(&[
                ("ObjectCount", Value::Int(3)),
                ("Distance", Value::Float(42.0)),
                ("Tail", Value::Int(0xEE)),
            ]),
            &EncodeOptions::default(),
        )
        .unwrap();
    assert!(frame.fd && frame.brs);
    assert_eq!(frame.data.len(), 64);
    assert_eq!(frame.data[63], 0xEE);
    let (m, decoded) = db.decode_frame(&frame, &DecodeOptions::default()).unwrap();
    assert_eq!(m.name, "RadarObjects");
    assert_eq!(decoded[2].1, Value::Int(0xEE));
    // A short payload fails unless truncation is allowed.
    assert!(radar
        .decode(&frame.data[..8], &DecodeOptions::default())
        .is_err());
    let partial = radar
        .decode(
            &frame.data[..8],
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
