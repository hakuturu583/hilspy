use super::encode::{AttEuler, PvtGeodetic};
use super::*;

fn pvt() -> Vec<u8> {
    PvtGeodetic {
        tow: Some(345_600_000),
        wnc: Some(2380),
        mode: 4,
        latitude: Some(35.6895f64.to_radians()),
        longitude: Some(139.6917f64.to_radians()),
        height: Some(40.0),
        vn: Some(1.5),
        ve: Some(-0.5),
        vu: Some(0.0),
        nr_sv: Some(18),
        h_accuracy: Some(2),
        ..Default::default()
    }
    .encode()
    .unwrap()
}

#[test]
fn crc_matches_xmodem() {
    assert_eq!(crc16(b"123456789"), 0x31C3);
}

#[test]
fn pvt_geodetic_round_trip() {
    let raw = pvt();
    assert_eq!(raw.len(), 96);
    let blocks = decode_all(&raw);
    assert_eq!(blocks.len(), 1);
    let b = &blocks[0];
    assert_eq!(
        (b.number, b.revision, b.name),
        (4007, 2, Some("PVTGeodetic"))
    );
    assert_eq!((b.tow, b.wnc), (Some(345_600_000), Some(2380)));
    assert_eq!(b.field("mode"), Some(&Field::Int(4)));
    assert_eq!(b.field("mode_name"), Some(&Field::Str("RTK Fixed".into())));
    assert_eq!(
        b.field("latitude"),
        Some(&Field::Float(35.6895f64.to_radians()))
    );
    assert_eq!(b.field("vn"), Some(&Field::Float(1.5)));
    assert_eq!(b.field("undulation"), Some(&Field::Null));
    assert_eq!(b.field("nr_sv"), Some(&Field::Int(18)));
    assert_eq!(b.raw, raw);
}

#[test]
fn att_euler_round_trip() {
    let raw = AttEuler {
        tow: Some(1000),
        wnc: Some(2380),
        mode: 4,
        heading: Some(90.25),
        pitch: Some(-1.0),
        roll: Some(0.5),
        ..Default::default()
    }
    .encode()
    .unwrap();
    let b = &decode_all(&raw)[0];
    assert_eq!(b.name, Some("AttEuler"));
    assert_eq!(b.field("heading"), Some(&Field::Float(90.25)));
    assert_eq!(b.field("heading_dot"), Some(&Field::Null));
    assert_eq!(
        b.field("mode"),
        Some(&Field::Str("HeadingPitchRollFixed".into()))
    );
}

#[test]
fn stream_in_chunks_with_noise() {
    let mut stream = b"$GPGGA,noise*00\r\n$".to_vec();
    let one = pvt();
    stream.extend_from_slice(&one);
    stream.extend_from_slice(b"\x00\x01$");
    let mut corrupt = one.clone();
    corrupt[20] ^= 0xFF;
    stream.extend_from_slice(&corrupt);
    stream.extend_from_slice(&one);
    let unknown = frame_block(4999, 0, &[1, 0, 0, 0, 2, 0, 9, 9]).unwrap();
    stream.extend_from_slice(&unknown);

    let mut decoder = Decoder::new();
    let mut blocks = Vec::new();
    for chunk in stream.chunks(7) {
        blocks.extend(decoder.feed(chunk));
    }
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].raw, one);
    assert_eq!(blocks[1].raw, one);
    assert_eq!(blocks[2].number, 4999);
    assert_eq!(blocks[2].name, None);
    assert_eq!(blocks[2].fields, None);
    assert_eq!((blocks[2].tow, blocks[2].wnc), (Some(1), Some(2)));
    let stats = decoder.stats();
    assert_eq!(stats.blocks, 3);
    assert_eq!(stats.crc_errors, 1);
    assert_eq!(decoder.pending(), 0);
}

#[test]
fn frame_block_pads_to_four() {
    let raw = frame_block(5000, 0, &[0; 7]).unwrap();
    assert_eq!(raw.len(), 16);
    assert_eq!(u16::from_le_bytes([raw[6], raw[7]]), 16);
    assert!(frame_block(0x2000, 0, &[]).is_err());
}
