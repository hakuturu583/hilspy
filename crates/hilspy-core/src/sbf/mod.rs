//! Septentrio Binary Format (SBF).
//!
//! [`Decoder`] splits a byte stream (serial port, TCP, file) into blocks,
//! checks their CRC and parses the bodies `libsbf` knows; [`frame_block`] and
//! the encoders in [`encode`] produce blocks, so a simulator can stand in for
//! a receiver.
//!
//! Block layout: `$@` sync, CRC-16/XMODEM (over everything after it), block
//! ID (bits 0-12 number, 13-15 revision), length (whole block, a multiple of
//! 4), then the body, which starts with TOW (u32, ms) and WNc (u16).

pub mod encode;
mod fields;

pub use fields::Field;

use crate::error::{Error, Result};

const SYNC: &[u8; 2] = b"$@";
const HEADER_LEN: usize = 8;

/// CRC-16/XMODEM (polynomial 0x1021, initial value 0), as SBF uses.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// One SBF block.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Block number (e.g. 4007 for PVTGeodetic).
    pub number: u16,
    pub revision: u8,
    /// `libsbf`'s name of the block, `None` for a block it does not parse.
    pub name: Option<&'static str>,
    /// GPS time of week in milliseconds (`None`: do-not-use value).
    pub tow: Option<u32>,
    /// GPS week number (`None`: do-not-use value).
    pub wnc: Option<u16>,
    /// The whole block, sync bytes to padding.
    pub raw: Vec<u8>,
    /// Parsed body fields, for blocks `libsbf` parses.
    pub fields: Option<Vec<(String, Field)>>,
    /// Why a block `libsbf` knows could not be parsed.
    pub parse_error: Option<String>,
}

impl Block {
    /// The body: everything after the 8-byte header.
    pub fn body(&self) -> &[u8] {
        &self.raw[HEADER_LEN..]
    }

    /// Build a block from one complete, CRC-checked block.
    fn from_raw(raw: Vec<u8>) -> Self {
        let id = u16::from_le_bytes([raw[4], raw[5]]);
        let number = id & 0x1FFF;
        let revision = (id >> 13) as u8;
        let body = &raw[HEADER_LEN..];
        let tow = body
            .get(0..4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .filter(|&t| t != u32::MAX);
        let wnc = body
            .get(4..6)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .filter(|&w| w != u16::MAX);
        let (name, fields, parse_error) = match libsbf::parse_datagram(&raw) {
            Ok(libsbf::Messages::Unsupported(_)) => (None, None, None),
            Ok(message) => (Some(message.type_name()), fields::fields(&message), None),
            Err(e) => (None, None, Some(format!("{e:?}"))),
        };
        Self {
            number,
            revision,
            name,
            tow,
            wnc,
            raw,
            fields,
            parse_error,
        }
    }

    /// A field by name.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields
            .as_ref()?
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }
}

/// Counters of a [`Decoder`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecoderStats {
    pub blocks: u64,
    pub crc_errors: u64,
    /// Bytes skipped while searching for a block (noise, NMEA, ASCII replies).
    pub skipped_bytes: u64,
}

/// A streaming SBF decoder: feed it bytes in any chunks, get whole blocks.
#[derive(Debug, Default)]
pub struct Decoder {
    buf: Vec<u8>,
    stats: DecoderStats,
}

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> DecoderStats {
        self.stats
    }

    /// Bytes held waiting for the rest of a block.
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Append bytes and return every block they complete.
    pub fn feed(&mut self, data: &[u8]) -> Vec<Block> {
        self.buf.extend_from_slice(data);
        let mut blocks = Vec::new();
        let mut pos = 0;
        loop {
            let Some(offset) = find_sync(&self.buf[pos..]) else {
                // Keep a trailing '$' that may start the next sync.
                let rest = &self.buf[pos..];
                let keep = usize::from(rest.last() == Some(&b'$'));
                let skip = rest.len() - keep;
                self.stats.skipped_bytes += skip as u64;
                pos += skip;
                break;
            };
            self.stats.skipped_bytes += offset as u64;
            pos += offset;
            let rest = &self.buf[pos..];
            if rest.len() < HEADER_LEN {
                break;
            }
            let length = usize::from(u16::from_le_bytes([rest[6], rest[7]]));
            if length < HEADER_LEN || length % 4 != 0 {
                // Not a block header: resync after this '$'.
                self.stats.skipped_bytes += 1;
                pos += 1;
                continue;
            }
            if rest.len() < length {
                break;
            }
            let crc = u16::from_le_bytes([rest[2], rest[3]]);
            if crc16(&rest[4..length]) != crc {
                self.stats.crc_errors += 1;
                self.stats.skipped_bytes += 1;
                pos += 1;
                continue;
            }
            blocks.push(Block::from_raw(rest[..length].to_vec()));
            self.stats.blocks += 1;
            pos += length;
        }
        self.buf.drain(..pos);
        blocks
    }
}

fn find_sync(data: &[u8]) -> Option<usize> {
    data.windows(2).position(|w| w == SYNC)
}

/// Decode a complete buffer (e.g. a recorded `.sbf` file).
pub fn decode_all(data: &[u8]) -> Vec<Block> {
    Decoder::new().feed(data)
}

/// Frame a block body: add the header and pad the body so the block length is
/// a multiple of 4. `body` starts with TOW and WNc.
pub fn frame_block(number: u16, revision: u8, body: &[u8]) -> Result<Vec<u8>> {
    if number > 0x1FFF || revision > 7 {
        return Err(Error::Sbf(format!(
            "block number {number} / revision {revision} out of range (13 / 3 bits)"
        )));
    }
    let padded = (HEADER_LEN + body.len()).div_ceil(4) * 4;
    let length = u16::try_from(padded)
        .map_err(|_| Error::Sbf(format!("a {padded}-byte block exceeds 65535 bytes")))?;
    let mut out = Vec::with_capacity(padded);
    out.extend_from_slice(SYNC);
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&(number | u16::from(revision) << 13).to_le_bytes());
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(body);
    out.resize(padded, 0);
    let crc = crc16(&out[4..]);
    out[2..4].copy_from_slice(&crc.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests;
