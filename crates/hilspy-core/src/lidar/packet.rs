//! Decoding data packets into points and encoding points into packets.

use super::proto::ReturnMode;
use super::{key, Model, TimeSpec};
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Packet};
use crate::time;

/// Points as columns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cloud {
    /// Metres: x forward, y left, z up.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub intensity: Vec<u8>,
    /// Channel (laser) index in packet order.
    pub channel: Vec<u16>,
    /// Degrees in [0, 360), clockwise from forward.
    pub azimuth: Vec<f32>,
    /// Metres.
    pub distance: Vec<f32>,
    /// Index into the return mode's returns.
    pub ret: Vec<u8>,
    /// Firing time, nanoseconds since the Unix epoch.
    pub time: Vec<i64>,
}

impl Cloud {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn append(&mut self, other: &mut Cloud) {
        self.x.append(&mut other.x);
        self.y.append(&mut other.y);
        self.z.append(&mut other.z);
        self.intensity.append(&mut other.intensity);
        self.channel.append(&mut other.channel);
        self.azimuth.append(&mut other.azimuth);
        self.distance.append(&mut other.distance);
        self.ret.append(&mut other.ret);
        self.time.append(&mut other.time);
    }

    /// Move the points from `at` on into a new cloud.
    pub fn split_off(&mut self, at: usize) -> Cloud {
        Cloud {
            x: self.x.split_off(at),
            y: self.y.split_off(at),
            z: self.z.split_off(at),
            intensity: self.intensity.split_off(at),
            channel: self.channel.split_off(at),
            azimuth: self.azimuth.split_off(at),
            distance: self.distance.split_off(at),
            ret: self.ret.split_off(at),
            time: self.time.split_off(at),
        }
    }
}

/// A decoded data packet.
#[derive(Clone, Debug)]
pub struct Decoded {
    /// The packet timestamp, nanoseconds since the Unix epoch.
    pub timestamp: i64,
    pub return_mode: &'static ReturnMode,
    /// Motor speed, revolutions per minute.
    pub rpm: Option<u16>,
    pub sequence: Option<u32>,
    pub cloud: Cloud,
    /// For each firing group in packet order: the index of its first point,
    /// its azimuth (degrees, without channel offsets) and firing time
    /// (nanoseconds since the Unix epoch).
    pub groups: Vec<(usize, f64, i64)>,
}

fn u16_at(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}

fn u32_at(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

impl Model {
    fn block_azimuth(&self, data: &[u8], base: usize) -> f64 {
        let coarse = u16_at(data, base + self.azimuth_offset);
        let raw = match self.fine_azimuth_offset {
            Some(o) => (u32::from(coarse) << 8) | u32::from(data[base + o]),
            None => coarse.into(),
        };
        (f64::from(raw) / self.azimuth_scale).rem_euclid(360.0)
    }

    fn write_azimuth(&self, data: &mut [u8], base: usize, azimuth: f64) {
        let turn = (360.0 * self.azimuth_scale).round() as u64;
        let raw = (azimuth.rem_euclid(360.0) * self.azimuth_scale).round() as u64 % turn;
        let o = base + self.azimuth_offset;
        match self.fine_azimuth_offset {
            Some(f) => {
                data[o..o + 2].copy_from_slice(&((raw >> 8) as u16).to_le_bytes());
                data[base + f] = (raw & 0xFF) as u8;
            }
            None => data[o..o + 2].copy_from_slice(&(raw as u16).to_le_bytes()),
        }
    }

    fn packet_time(&self, data: &[u8], received: i64) -> i64 {
        match self.time {
            TimeSpec::None => received,
            TimeSpec::DateTime {
                seconds: o,
                year_base,
                micros,
            } => time::unix_seconds(
                year_base + i64::from(data[o]),
                data[o + 1],
                data[o + 2],
                data[o + 3],
                data[o + 4],
                data[o + 5],
            )
            .map_or(received, |s| {
                s * 1_000_000_000 + i64::from(u32_at(data, micros)) * 1000
            }),
            TimeSpec::UnixSeconds { seconds: o, micros } => {
                let s = data[o + 1..o + 6]
                    .iter()
                    .fold(0u64, |acc, b| (acc << 8) | u64::from(*b));
                s as i64 * 1_000_000_000 + i64::from(u32_at(data, micros)) * 1000
            }
            TimeSpec::TopOfHour { micros } => {
                let us = u32_at(data, micros);
                if us >= 3_600_000_000 {
                    return received;
                }
                // The hour of the reception time; the packet may be from
                // just before or after an hour boundary.
                const HOUR: i64 = 3_600_000_000_000;
                let t = received.div_euclid(HOUR) * HOUR + i64::from(us) * 1000;
                if t - received > HOUR / 2 {
                    t - HOUR
                } else if received - t > HOUR / 2 {
                    t + HOUR
                } else {
                    t
                }
            }
        }
    }

    fn write_time(&self, data: &mut [u8], t: f64) -> Result<()> {
        let mut secs = t.floor() as i64;
        let mut us = ((t - t.floor()) * 1e6).round() as u32;
        if us >= 1_000_000 {
            secs += 1;
            us -= 1_000_000;
        }
        match self.time {
            TimeSpec::None => {}
            TimeSpec::DateTime {
                seconds: o,
                year_base,
                micros,
            } => {
                let (y, mo, d, h, mi, s) = time::utc(secs);
                let year = u8::try_from(y - year_base).map_err(|_| {
                    Error::encode(format!(
                        "{}: year {y} does not fit the packet (base {year_base})",
                        self.name()
                    ))
                })?;
                data[o..o + 6].copy_from_slice(&[year, mo, d, h, mi, s]);
                data[micros..micros + 4].copy_from_slice(&us.to_le_bytes());
            }
            TimeSpec::UnixSeconds { seconds: o, micros } => {
                let s = u64::try_from(secs)
                    .ok()
                    .filter(|s| *s < 1 << 40)
                    .ok_or_else(|| {
                        Error::encode(format!("{}: time {t} out of range", self.name()))
                    })?;
                data[o] = 0;
                data[o + 1..o + 6].copy_from_slice(&s.to_be_bytes()[3..]);
                data[micros..micros + 4].copy_from_slice(&us.to_le_bytes());
            }
            TimeSpec::TopOfHour { micros } => {
                let us = (t.rem_euclid(3600.0) * 1e6).round() as u64 % 3_600_000_000;
                data[micros..micros + 4].copy_from_slice(&(us as u32).to_le_bytes());
            }
        }
        Ok(())
    }

    /// Decode a data packet. `received` (reception time) dates packets
    /// without a full timestamp; it defaults to now.
    pub fn decode(&self, data: &[u8], received: Option<f64>) -> Result<Decoded> {
        let name = self.name();
        if data.len() < self.min_len {
            return Err(Error::decode(format!(
                "a {}-byte packet is not a {name} data packet (at least {} bytes)",
                data.len(),
                self.min_len
            )));
        }
        for (o, v, verify) in &self.constants {
            if *verify && &data[*o..o + v.len()] != *v {
                return Err(Error::decode(format!(
                    "not a {name} data packet: bytes {o}.. are {:02x?}, not {:02x?}",
                    &data[*o..o + v.len()],
                    v
                )));
            }
        }
        let modes = &self.profile.return_modes;
        let return_mode = match self.return_mode_offset {
            None => &modes[0],
            Some(o) => modes
                .iter()
                .find(|m| m.code == u32::from(data[o]))
                .ok_or_else(|| {
                    Error::decode(format!(
                        "{name}: unknown return mode 0x{:02X} (known: {})",
                        data[o],
                        modes
                            .iter()
                            .map(|m| format!("0x{:02X} {}", m.code, m.name))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?,
        };
        let returns = return_mode.returns.len();
        let per_group = self.banks * returns;
        if self.block_count % per_group != 0 {
            return Err(Error::decode(format!(
                "{name}: {} blocks do not hold whole firings of {returns} returns",
                self.block_count
            )));
        }
        let groups = self.block_count / per_group;
        let fpb = self.firings_per_block;
        let firings = (groups * fpb) as i64;
        let received = (received.unwrap_or_else(now_timestamp) * 1e9).round() as i64;
        let timestamp = self.packet_time(data, received);
        let rpm = self
            .motor_speed_offset
            .map(|o| u16_at(data, o))
            .filter(|r| *r > 0);
        let period = self.period_ns as f64 * 1e-9;
        let default_step = 6.0 * rpm.map_or(self.profile.rpm, f64::from) * period * fpb as f64;
        let azimuths: Vec<f64> = (0..groups)
            .map(|g| {
                let base = self.blocks_offset + g * per_group * self.block_size;
                self.block_azimuth(data, base)
            })
            .collect();
        // Azimuth advance from one group to the next, for firings that share
        // a block (VLP-16): the neighbouring group's, if plausible.
        let step = |g: usize| {
            let d = if g + 1 < groups {
                (azimuths[g + 1] - azimuths[g]).rem_euclid(360.0)
            } else if g > 0 {
                (azimuths[g] - azimuths[g - 1]).rem_euclid(360.0)
            } else {
                default_step
            };
            if d > 0.0 && d < 4.0 * default_step.max(0.01) {
                d
            } else {
                default_step
            }
        };
        let first = if self.from_end { firings - 1 } else { 0 };
        let firing_time =
            |i: usize| timestamp + self.offset_ns + self.period_ns * (i as i64 - first);
        let (min, max) = (self.profile.min_range, self.profile.max_range);
        let mut cloud = Cloud::default();
        let mut group_starts = Vec::with_capacity(groups);
        for (g, &azimuth) in azimuths.iter().enumerate() {
            group_starts.push((cloud.len(), azimuth, firing_time(g * fpb)));
            let advance = step(g) / fpb as f64;
            for r in 0..returns {
                for bank in 0..self.banks {
                    let base = self.blocks_offset
                        + (g * per_group + r * self.banks + bank) * self.block_size;
                    if !self.flags.is_empty()
                        && u16_at(data, base) != self.flags[bank % self.flags.len()]
                    {
                        continue;
                    }
                    for f in 0..fpb {
                        let az = azimuth + advance * f as f64;
                        let t = firing_time(g * fpb + f);
                        for u in 0..self.units_per_firing {
                            let o = base
                                + self.units_offset
                                + (f * self.units_per_firing + u) * self.unit_size;
                            let raw = u16_at(data, o);
                            if raw == 0 {
                                continue;
                            }
                            let d = f64::from(raw) * self.distance_unit;
                            if d < min || d > max {
                                continue;
                            }
                            let c = bank * self.units_per_firing + u;
                            let ch = &self.channels[c];
                            let a = (az + ch.azimuth_offset).rem_euclid(360.0);
                            let (sin_a, cos_a) = a.to_radians().sin_cos();
                            let xy = d * ch.cos_el;
                            cloud.x.push((xy * cos_a) as f32);
                            cloud.y.push((-xy * sin_a) as f32);
                            cloud.z.push((d * ch.sin_el) as f32);
                            cloud.intensity.push(data[o + 2]);
                            cloud.channel.push(c as u16);
                            cloud.azimuth.push(a as f32);
                            cloud.distance.push(d as f32);
                            cloud.ret.push(r as u8);
                            cloud.time.push(t + ch.firing_ns);
                        }
                    }
                }
            }
        }
        Ok(Decoded {
            timestamp,
            return_mode,
            rpm,
            sequence: self.sequence_offset.map(|o| u32_at(data, o)),
            cloud,
            groups: group_starts,
        })
    }
}

/// Points to encode.
#[derive(Clone, Debug, PartialEq)]
pub enum Points {
    /// Metres: x forward, y left, z up. Each point goes to the channel of
    /// the nearest elevation.
    Cartesian {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
    },
    /// Distance (metres), azimuth (degrees, with the channel's offset) and
    /// channel index, as decoding reports them.
    Polar {
        distance: Vec<f64>,
        azimuth: Vec<f64>,
        channel: Vec<f64>,
    },
}

/// Encodes point clouds into the packets of one revolution, in firing
/// order. Each firing column keeps the nearest point per channel.
#[derive(Debug)]
pub struct Encoder {
    rpm: f64,
    return_code: u8,
    sequence: u32,
}

impl Encoder {
    /// `rpm` defaults to the profile's; `return_mode` (a single-return
    /// mode's name) to `strongest`, or the first single-return mode.
    pub fn new(model: &Model, rpm: Option<f64>, return_mode: Option<&str>) -> Result<Self> {
        let rpm = rpm.unwrap_or(model.profile.rpm);
        if !(rpm.is_finite() && rpm > 0.0) {
            return Err(Error::Config(format!("rpm must be positive, not {rpm}")));
        }
        let modes = &model.profile.return_modes;
        let mode = match return_mode {
            Some(name) => modes
                .iter()
                .find(|m| key(&m.name) == key(name))
                .ok_or_else(|| {
                    Error::Config(format!(
                        "{} has no return mode {name:?} (known: {})",
                        model.name(),
                        modes
                            .iter()
                            .map(|m| m.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?,
            None => modes
                .iter()
                .find(|m| m.name == "strongest")
                .or_else(|| modes.iter().find(|m| m.returns.len() == 1))
                .unwrap_or(&modes[0]),
        };
        if mode.returns.len() != 1 {
            return Err(Error::Config(format!(
                "encoding writes single-return packets; {:?} has {} returns",
                mode.name,
                mode.returns.len()
            )));
        }
        Ok(Self {
            rpm,
            return_code: mode.code as u8,
            sequence: 0,
        })
    }

    /// Firing columns of one revolution (or of the azimuth range), the
    /// first column's azimuth, the step between them (degrees) and whether
    /// they make a full turn.
    pub(crate) fn columns(&self, model: &Model) -> Result<(usize, f64, f64, bool)> {
        let p = model.profile;
        let start = p.azimuth_start;
        let mut span = (p.azimuth_end - start).rem_euclid(360.0);
        let full = span == 0.0;
        if full {
            span = 360.0;
        }
        let step = 6.0 * self.rpm * model.period_ns as f64 * 1e-9;
        // Whole packets per revolution, as a sensor streaming at a constant
        // rate approximately sends: the step stretches slightly to fit.
        let per_packet = (model.block_count / model.banks * model.firings_per_block) as f64;
        let n = ((span / step / per_packet).round() * per_packet).max(per_packet);
        if n * model.channels.len() as f64 > 5e7 {
            return Err(Error::Config(format!(
                "{} at {} rpm makes too many firings per revolution",
                model.name(),
                self.rpm
            )));
        }
        Ok((n as usize, start, span / n, full))
    }

    /// The packets of one revolution starting at `timestamp` (seconds since
    /// the Unix epoch), with their timestamps.
    pub fn encode(
        &mut self,
        model: &Model,
        points: &Points,
        intensity: Option<&[f64]>,
        timestamp: f64,
    ) -> Result<Vec<Packet>> {
        let name = model.name();
        let len = match points {
            Points::Cartesian { x, y, z } => {
                if y.len() != x.len() || z.len() != x.len() {
                    return Err(Error::encode(format!(
                        "{name}: x, y and z have {}, {} and {} points",
                        x.len(),
                        y.len(),
                        z.len()
                    )));
                }
                x.len()
            }
            Points::Polar {
                distance,
                azimuth,
                channel,
            } => {
                if azimuth.len() != distance.len() || channel.len() != distance.len() {
                    return Err(Error::encode(format!(
                        "{name}: distance, azimuth and channel have {}, {} and {} points",
                        distance.len(),
                        azimuth.len(),
                        channel.len()
                    )));
                }
                distance.len()
            }
        };
        if let Some(i) = intensity {
            if i.len() != len && i.len() != 1 {
                return Err(Error::encode(format!(
                    "{name}: {len} points but {} intensities",
                    i.len()
                )));
            }
        }
        let (n, start, step, full) = self.columns(model)?;
        let nch = model.channels.len();
        let mut by_elevation: Vec<(f64, usize)> = model
            .channels
            .iter()
            .enumerate()
            .map(|(i, c)| (c.elevation, i))
            .collect();
        by_elevation.sort_by(|a, b| a.0.total_cmp(&b.0));
        let nearest = |el: f64| {
            let i = by_elevation.partition_point(|(e, _)| *e < el);
            match (i.checked_sub(1), by_elevation.get(i)) {
                (Some(a), Some(b)) if el - by_elevation[a].0 <= b.0 - el => by_elevation[a].1,
                (_, Some(b)) => b.1,
                (Some(a), None) => by_elevation[a].1,
                (None, None) => unreachable!("a model has channels"),
            }
        };
        let mut distance = vec![0u16; n * nch];
        let mut intensities = vec![0u8; n * nch];
        let (min, max) = (model.profile.min_range, model.profile.max_range);
        for i in 0..len {
            let (c, az, d) = match points {
                Points::Cartesian { x, y, z } => {
                    let (x, y, z) = (x[i], y[i], z[i]);
                    let d = (x * x + y * y + z * z).sqrt();
                    if d.is_nan() || d <= 0.0 {
                        continue;
                    }
                    let c = nearest((z / d).asin().to_degrees());
                    (c, (-y).atan2(x).to_degrees(), d)
                }
                Points::Polar {
                    distance,
                    azimuth,
                    channel,
                } => {
                    let c = channel[i];
                    if !(c >= 0.0 && c.fract() == 0.0 && (c as usize) < nch) {
                        return Err(Error::encode(format!(
                            "{name}: channel {c} is not in 0..{nch}"
                        )));
                    }
                    (c as usize, azimuth[i], distance[i])
                }
            };
            if !(d >= min && d <= max) || !az.is_finite() {
                continue;
            }
            let raw = (d / model.distance_unit).round();
            if !(1.0..=65535.0).contains(&raw) {
                continue;
            }
            let rel = (az - model.channels[c].azimuth_offset - start).rem_euclid(360.0);
            let mut k = (rel / step).round() as usize;
            if k >= n {
                if !full {
                    continue;
                }
                k %= n;
            }
            let slot = k * nch + c;
            if distance[slot] == 0 || (raw as u16) < distance[slot] {
                distance[slot] = raw as u16;
                intensities[slot] = intensity.map_or(0, |v| {
                    // One intensity applies to every point.
                    v[if v.len() == 1 { 0 } else { i }]
                        .round()
                        .clamp(0.0, 255.0) as u8
                });
            }
        }

        let fpb = model.firings_per_block;
        let groups = model.block_count / model.banks;
        let per_packet = groups * fpb;
        let period = model.period_ns as f64 * 1e-9;
        let first = if model.from_end { per_packet - 1 } else { 0 };
        let mut packets = Vec::with_capacity(n.div_ceil(per_packet));
        for p in 0..n.div_ceil(per_packet) {
            let mut data = vec![0u8; model.packet_size];
            for (o, v, _) in &model.constants {
                data[*o..o + v.len()].copy_from_slice(v);
            }
            for g in 0..groups {
                let j0 = p * per_packet + g * fpb;
                for bank in 0..model.banks {
                    let base = model.blocks_offset + (g * model.banks + bank) * model.block_size;
                    if !model.flags.is_empty() {
                        let flag = model.flags[bank % model.flags.len()];
                        data[base..base + 2].copy_from_slice(&flag.to_le_bytes());
                    }
                    model.write_azimuth(&mut data, base, start + j0 as f64 * step);
                    for f in 0..fpb {
                        let j = j0 + f;
                        if j >= n {
                            continue;
                        }
                        for u in 0..model.units_per_firing {
                            let slot = j * nch + bank * model.units_per_firing + u;
                            let o = base
                                + model.units_offset
                                + (f * model.units_per_firing + u) * model.unit_size;
                            data[o..o + 2].copy_from_slice(&distance[slot].to_le_bytes());
                            data[o + 2] = intensities[slot];
                        }
                    }
                }
            }
            if let Some(o) = model.return_mode_offset {
                data[o] = self.return_code;
            }
            if let Some(o) = model.motor_speed_offset {
                let rpm = self.rpm.round().min(65535.0) as u16;
                data[o..o + 2].copy_from_slice(&rpm.to_le_bytes());
            }
            let t = timestamp + (p * per_packet) as f64 * period - model.offset_ns as f64 * 1e-9
                + first as f64 * period;
            model.write_time(&mut data, t)?;
            if let Some(o) = model.sequence_offset {
                data[o..o + 4].copy_from_slice(&self.sequence.to_le_bytes());
                self.sequence = self.sequence.wrapping_add(1);
            }
            packets.push(Packet {
                data,
                timestamp: Some(t),
                ..Packet::default()
            });
        }
        Ok(packets)
    }
}
