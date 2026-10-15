//! Bit layout of DBC signals.
//!
//! Little endian (Intel, `@1`): the start bit is the least significant bit;
//! bit `k` of the value sits at payload bit `start + k`, counting bit 0 of
//! byte 0 as 0.
//!
//! Big endian (Motorola, `@0`): the start bit is the most significant bit,
//! numbered in the same per-byte "sawtooth" (byte * 8 + bit, bit 0 = LSB). In
//! a most-significant-first linear numbering (bit 7 of byte 0 is 0) the signal
//! is the `size` consecutive bits from `msb_index(start)`.

use super::ByteOrder;

/// Position of a sawtooth bit in most-significant-first linear numbering.
fn msb_index(start: u32) -> u32 {
    (start / 8) * 8 + (7 - start % 8)
}

/// The payload bits a signal occupies, least significant value bit first,
/// as `(byte, bit)` pairs; `None` if they do not fit in `len` bytes.
fn positions(start: u32, size: u32, order: ByteOrder, len: usize) -> Option<Vec<(usize, u32)>> {
    let total = len as u64 * 8;
    let mut out = Vec::with_capacity(size as usize);
    match order {
        ByteOrder::LittleEndian => {
            if u64::from(start) + u64::from(size) > total {
                return None;
            }
            for k in 0..size {
                let b = start + k;
                out.push(((b / 8) as usize, b % 8));
            }
        }
        ByteOrder::BigEndian => {
            let first = msb_index(start);
            if u64::from(first) + u64::from(size) > total {
                return None;
            }
            // Value bit k (LSB = 0) is linear bit first + size - 1 - k.
            for k in 0..size {
                let l = first + size - 1 - k;
                out.push(((l / 8) as usize, 7 - l % 8));
            }
        }
    }
    Some(out)
}

/// Whether the signal fits in a payload of `len` bytes.
pub(crate) fn fits(start: u32, size: u32, order: ByteOrder, len: usize) -> bool {
    (1..=64).contains(&size) && positions(start, size, order, len).is_some()
}

/// The raw (unsigned) value of a signal, or `None` if it does not fit in `data`.
pub(crate) fn read(data: &[u8], start: u32, size: u32, order: ByteOrder) -> Option<u64> {
    let mut raw = 0u64;
    for (k, (byte, bit)) in positions(start, size, order, data.len())?
        .into_iter()
        .enumerate()
    {
        if data[byte] >> bit & 1 == 1 {
            raw |= 1 << k;
        }
    }
    Some(raw)
}

/// Store the low `size` bits of `raw`; `false` if the signal does not fit.
pub(crate) fn write(data: &mut [u8], start: u32, size: u32, order: ByteOrder, raw: u64) -> bool {
    let Some(pos) = positions(start, size, order, data.len()) else {
        return false;
    };
    for (k, (byte, bit)) in pos.into_iter().enumerate() {
        if raw >> k & 1 == 1 {
            data[byte] |= 1 << bit;
        } else {
            data[byte] &= !(1 << bit);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_endian_crosses_bytes() {
        let mut data = [0u8; 8];
        assert!(write(&mut data, 4, 12, ByteOrder::LittleEndian, 0xABC));
        assert_eq!(&data[..3], &[0xC0, 0xAB, 0x00]);
        assert_eq!(read(&data, 4, 12, ByteOrder::LittleEndian), Some(0xABC));
    }

    #[test]
    fn big_endian_sawtooth() {
        // A 16-bit Motorola signal starting at bit 7 is bytes 0..2, MSB first.
        let mut data = [0u8; 8];
        assert!(write(&mut data, 7, 16, ByteOrder::BigEndian, 0x1234));
        assert_eq!(&data[..2], &[0x12, 0x34]);
        // A 12-bit signal starting at bit 3: low nibble of byte 0, all of byte 1.
        let mut data = [0u8; 8];
        assert!(write(&mut data, 3, 12, ByteOrder::BigEndian, 0xABC));
        assert_eq!(&data[..2], &[0x0A, 0xBC]);
        assert_eq!(read(&data, 3, 12, ByteOrder::BigEndian), Some(0xABC));
    }

    #[test]
    fn out_of_range() {
        assert!(!fits(60, 8, ByteOrder::LittleEndian, 8));
        assert!(fits(56, 8, ByteOrder::LittleEndian, 8));
        // Motorola starting at bit 0 of the last byte runs past the end.
        assert!(!fits(56, 2, ByteOrder::BigEndian, 8));
        assert!(fits(63, 8, ByteOrder::BigEndian, 8));
    }
}
