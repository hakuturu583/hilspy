use super::*;

fn model(name: &str) -> Model {
    Model::find(name, None).unwrap()
}

/// Polar points: every channel at `columns` azimuths, distances by channel.
fn grid(m: &Model, columns: &[f64]) -> (Points, Vec<(u16, f64, f64)>) {
    let (mut distance, mut azimuth, mut channel) = (Vec::new(), Vec::new(), Vec::new());
    let mut expect = Vec::new();
    for (k, base) in columns.iter().enumerate() {
        for (c, ch) in m.channels.iter().enumerate() {
            let d =
                (m.profile.min_range + 1.0 + c as f64 * 0.1 + k as f64).min(m.profile.max_range);
            let a = (base + ch.azimuth_offset).rem_euclid(360.0);
            distance.push(d);
            azimuth.push(a);
            channel.push(c as f64);
            expect.push((c as u16, a, d));
        }
    }
    (
        Points::Polar {
            distance,
            azimuth,
            channel,
        },
        expect,
    )
}

fn decode_all(m: &Model, packets: &[crate::Packet]) -> Cloud {
    let mut cloud = Cloud::default();
    for p in packets {
        assert_eq!(p.data.len(), m.packet_size());
        let mut d = m.decode(&p.data, p.timestamp).unwrap();
        cloud.append(&mut d.cloud);
    }
    cloud
}

#[test]
fn every_profile_round_trips() {
    for name in names() {
        let m = model(&name);
        let mut enc = Encoder::new(&m, None, None).unwrap();
        let (_, _, step, _) = enc.columns(&m).unwrap();
        let start = m.profile.azimuth_start;
        // Column azimuths: on the encoder's grid, inside the azimuth range.
        let columns: Vec<f64> = [3usize, 10, 101]
            .iter()
            .map(|k| start + *k as f64 * step)
            .collect();
        let (points, expect) = grid(&m, &columns);
        let t0 = 1_760_000_000.25;
        let packets = enc.encode(&m, &points, None, t0).unwrap();
        let cloud = decode_all(&m, &packets);
        assert_eq!(cloud.len(), expect.len(), "{name}");
        let mut got: Vec<_> = (0..cloud.len())
            .map(|i| (cloud.channel[i], cloud.azimuth[i], cloud.distance[i]))
            .collect();
        got.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap());
        let mut want = expect.clone();
        want.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.0, w.0, "{name}");
            let da = (f64::from(g.1) - w.1 + 180.0).rem_euclid(360.0) - 180.0;
            // Azimuths are quantized to the packet's resolution; a firing
            // sharing a block (VLP-16) is interpolated.
            assert!(da.abs() < 0.02, "{name}: azimuth {} vs {}", g.1, w.1);
            assert!((f64::from(g.2) - w.2).abs() <= m.distance_unit, "{name}");
        }
        // Firing times follow the encoded scan start.
        let first = *cloud.time.iter().min().unwrap() as f64 * 1e-9;
        assert!(
            (first - (t0 + 3.0 * step / (6.0 * m.profile.rpm))).abs() < 2e-4,
            "{name}: {first}"
        );
        // Points lie on the unit sphere scaled by distance.
        for i in 0..cloud.len() {
            let r = (cloud.x[i].powi(2) + cloud.y[i].powi(2) + cloud.z[i].powi(2)).sqrt();
            assert!((r - cloud.distance[i]).abs() < 1e-3, "{name}");
        }
    }
}

#[test]
fn a_revolution_fills_whole_packets() {
    let m = model("pandar_xt32");
    let mut enc = Encoder::new(&m, None, None).unwrap();
    let packets = enc
        .encode(
            &m,
            &Points::Cartesian {
                x: vec![],
                y: vec![],
                z: vec![],
            },
            None,
            0.0,
        )
        .unwrap();
    // 0.18 degrees per firing at 600 rpm, 8 firings per packet.
    assert_eq!(packets.len(), 250);
    let seq: Vec<u32> = packets
        .iter()
        .map(|p| m.decode(&p.data, None).unwrap().sequence.unwrap())
        .collect();
    assert_eq!(seq, (0..250).collect::<Vec<_>>());
    let d = m.decode(&packets[1].data, None).unwrap();
    assert_eq!(d.rpm, Some(600));
    assert_eq!(d.return_mode.name, "strongest");
    assert!((d.groups[0].1 - 8.0 * 0.18).abs() < 1e-9);
}

#[test]
fn cartesian_points_go_to_the_nearest_elevation() {
    let m = model("vlp16");
    let mut enc = Encoder::new(&m, None, None).unwrap();
    // 10 m ahead at +1 degree (channel 1) and 20 m to the left at -15 (channel 0).
    let e1 = 1f64.to_radians();
    let e0 = (-15f64).to_radians();
    let points = Points::Cartesian {
        x: vec![10.0 * e1.cos(), 0.0],
        y: vec![0.0, 20.0 * e0.cos()],
        z: vec![10.0 * e1.sin(), 20.0 * e0.sin()],
    };
    let packets = enc.encode(&m, &points, Some(&[7.0, 300.0]), 0.0).unwrap();
    let c = decode_all(&m, &packets);
    assert_eq!(c.channel, vec![1, 0]);
    assert_eq!(c.intensity, vec![7, 255]);
    assert!((c.x[0] - 9.998).abs() < 0.01 && c.y[0].abs() < 0.01);
    assert!((c.azimuth[1] - 270.0).abs() < 0.02, "{}", c.azimuth[1]);
    assert!((c.y[1] - 19.319).abs() < 0.01, "{}", c.y[1]);
}

fn velodyne_packet(azimuths: [u16; 12], distance: u16, mode: u8, micros: u32) -> Vec<u8> {
    let mut p = vec![0u8; 1206];
    for (b, az) in azimuths.iter().enumerate() {
        let o = b * 100;
        p[o..o + 2].copy_from_slice(&[0xFF, 0xEE]);
        p[o + 2..o + 4].copy_from_slice(&az.to_le_bytes());
        for u in 0..32 {
            p[o + 4 + u * 3..o + 6 + u * 3].copy_from_slice(&distance.to_le_bytes());
            p[o + 6 + u * 3] = u as u8;
        }
    }
    p[1200..1204].copy_from_slice(&micros.to_le_bytes());
    p[1204] = mode;
    p[1205] = 0x22;
    p
}

#[test]
fn decodes_a_vlp16_packet() {
    let m = model("VLP-16");
    let az: [u16; 12] = std::array::from_fn(|b| 35_990 + 40 * b as u16);
    let p = velodyne_packet(az.map(|a| a % 36_000), 5000, 0x37, 1_500_000);
    // Received at 10:00:02 UTC on some day: the packet is 1.5 s past 10:00.
    let hour = 1_760_000_400.0 - 1_760_000_400f64 % 3600.0;
    let d = m.decode(&p, Some(hour + 2.0)).unwrap();
    assert_eq!(d.timestamp, ((hour + 1.5) * 1e9) as i64);
    assert_eq!(d.cloud.len(), 12 * 32);
    // Channel 0: -15 degrees, 10 m, azimuth 359.9.
    let a = 359.9f64.to_radians();
    let e = (-15f64).to_radians();
    assert_eq!(d.cloud.channel[0], 0);
    assert!((f64::from(d.cloud.x[0]) - 10.0 * e.cos() * a.cos()).abs() < 1e-4);
    assert!((f64::from(d.cloud.y[0]) + 10.0 * e.cos() * a.sin()).abs() < 1e-4);
    assert!((f64::from(d.cloud.z[0]) - 10.0 * e.sin()).abs() < 1e-4);
    // The second firing of a block is half way to the next block, across 0.
    assert_eq!(d.cloud.channel[16], 0);
    assert!(
        (d.cloud.azimuth[16] - 0.1).abs() < 1e-4,
        "{}",
        d.cloud.azimuth[16]
    );
    // Channel 1 fires 2.304 us after channel 0; firings are 55.296 us apart.
    assert_eq!(d.cloud.time[1] - d.cloud.time[0], 2304);
    assert_eq!(d.cloud.time[16] - d.cloud.time[0], 55_296);
    assert_eq!(d.cloud.intensity[5], 5);

    // Dual return: blocks pair up (last, strongest) at one azimuth.
    let az: [u16; 12] = std::array::from_fn(|b| 100 + 40 * (b / 2) as u16);
    let d = m
        .decode(&velodyne_packet(az, 5000, 0x39, 0), Some(hour))
        .unwrap();
    assert_eq!(d.return_mode.returns, ["last", "strongest"]);
    assert_eq!(d.groups.len(), 6);
    assert_eq!(&d.cloud.ret[..64].iter().filter(|r| **r == 1).count(), &32);
}

#[test]
fn decodes_a_hesai_header_and_date() {
    let m = model("xt32");
    let mut enc = Encoder::new(&m, Some(1200.0), Some("last")).unwrap();
    // 2025-10-09 08:00:00.123456 UTC
    let t = 1_759_996_800.123_456;
    let p = &enc
        .encode(
            &m,
            &Points::Cartesian {
                x: vec![],
                y: vec![],
                z: vec![],
            },
            None,
            t,
        )
        .unwrap()[0];
    let b = &p.data;
    assert_eq!(&b[..2], &[0xEE, 0xFF]);
    assert_eq!((b[6], b[7]), (32, 8));
    assert_eq!(b[1062], 0x38);
    assert_eq!(u16::from_le_bytes([b[1063], b[1064]]), 1200);
    assert_eq!(&b[1065..1071], &[125, 10, 9, 8, 0, 0]);
    let d = m.decode(b, None).unwrap();
    assert_eq!(d.rpm, Some(1200));
    // The last firing of the packet is 5.632 us after its timestamp.
    // The last of the packet's 8 firings is 5.632 us after the packet
    // timestamp: t + 7 x 50 us - 5.632 us, in whole microseconds.
    assert_eq!(d.timestamp, 1_759_996_800_123_800_000);
}

#[test]
fn rejects_other_packets() {
    let m = model("pandar_xt32");
    assert!(m.decode(&[0u8; 100], None).is_err());
    let mut enc = Encoder::new(&m, None, None).unwrap();
    let mut p = enc
        .encode(
            &m,
            &Points::Cartesian {
                x: vec![],
                y: vec![],
                z: vec![],
            },
            None,
            0.0,
        )
        .unwrap()
        .remove(0)
        .data;
    p[1062] = 0x99;
    assert!(m
        .decode(&p, None)
        .unwrap_err()
        .to_string()
        .contains("return mode 0x99"));
    p[1062] = 0x37;
    p[7] = 6;
    assert!(m
        .decode(&p, None)
        .unwrap_err()
        .to_string()
        .contains("not a pandar_xt32"));
    assert!(Encoder::new(&m, None, Some("last_strongest")).is_err());
    assert!(Encoder::new(&m, Some(0.0), None).is_err());
    assert!(Model::find("nope", None).is_err());
}

#[test]
fn names_and_aliases() {
    for (alias, name) in [
        ("VLP-16", "vlp16"),
        ("Pandar XT-32", "pandar_xt32"),
        ("OT128", "pandar_128e4x"),
        ("PandarAT128E2X", "pandar_at128"),
    ] {
        assert_eq!(model(alias).name(), name);
    }
}

#[test]
fn invalid_profiles_are_rejected() {
    let mut p = defaults()[0].clone();
    p.channels.pop();
    let p: &'static Profile = Box::leak(Box::new(p));
    assert!(Model::new(p).unwrap_err().to_string().contains("channels"));
    let mut p = defaults()[0].clone();
    p.layout.as_mut().unwrap().packet_size = 100;
    let p: &'static Profile = Box::leak(Box::new(p));
    assert!(Model::new(p).unwrap_err().to_string().contains("after the"));
}
