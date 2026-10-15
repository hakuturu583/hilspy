//! LiDAR data packets described by profiles.
//!
//! A profile ([`proto::Profile`]) says how a sensor lays out its UDP data
//! packets (blocks of laser returns, azimuth, return mode, timestamp) and
//! how its channels are aimed. The default profiles (Hesai Pandar/QT/XT/AT,
//! Velodyne VLP/HDL/VLS) are a protobuf `ProfileSet` embedded from
//! `lidar/profiles.binpb`, compiled from `lidar/profiles/*.txtpb`; more can
//! be loaded from any file in that format ([`load`]).
//!
//! [`Model`] is a validated profile: it decodes packets into points
//! ([`Model::decode`]) and encodes a point cloud into the packets of one
//! revolution ([`Model::encode`]).

mod packet;
mod time;

use std::path::Path;
use std::sync::OnceLock;

use prost::Message as _;

use crate::error::{Error, Result};

pub use packet::{Cloud, Decoded, Encoder, Points};

/// The profile schema (`lidar/lidar_profile.proto`).
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/hilspy.lidar.rs"));
}

use proto::timestamp::Kind;
use proto::Profile;

static DEFAULT_PROFILES: &[u8] = include_bytes!("../../lidar/profiles.binpb");

/// The embedded default profiles.
pub fn defaults() -> &'static [Profile] {
    static PROFILES: OnceLock<Vec<Profile>> = OnceLock::new();
    PROFILES.get_or_init(|| {
        proto::ProfileSet::decode(DEFAULT_PROFILES)
            .expect("the embedded LiDAR profiles do not decode")
            .profiles
    })
}

/// Names of the default profiles, in file order.
pub fn names() -> Vec<String> {
    defaults().iter().map(|p| p.name.clone()).collect()
}

/// Profiles from a serialized `ProfileSet` file.
pub fn load(path: impl AsRef<Path>) -> Result<Vec<Profile>> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Config(format!("LiDAR profiles {}: {e}", path.display())))?;
    proto::ProfileSet::decode(bytes.as_slice())
        .map(|s| s.profiles)
        .map_err(|e| {
            Error::Config(format!(
                "{} is not a serialized hilspy.lidar.ProfileSet: {e}",
                path.display()
            ))
        })
}

/// Names compare without case, `-`, `_`, `.` or spaces (`VLP-16` = `vlp16`).
fn key(name: &str) -> String {
    name.chars()
        .filter(|c| !"-_. ".contains(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// The profile a name or alias refers to: `extra` first, then the defaults.
pub fn find(name: &str, extra: &[Profile]) -> Result<&'static Profile> {
    let k = key(name);
    let matches = |p: &Profile| {
        key(&p.name) == k || key(&p.model) == k || p.aliases.iter().any(|a| key(a) == k)
    };
    // Extra profiles are leaked so every Model can borrow a 'static profile;
    // they are loaded once per codec and are small.
    if let Some(p) = extra.iter().find(|p| matches(p)) {
        return Ok(Box::leak(Box::new(p.clone())));
    }
    defaults().iter().find(|p| matches(p)).ok_or_else(|| {
        let known: Vec<_> = extra
            .iter()
            .chain(defaults())
            .map(|p| p.name.as_str())
            .collect();
        Error::Config(format!(
            "unknown LiDAR {name:?} (known: {})",
            known.join(", ")
        ))
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ChannelGeom {
    pub elevation: f64,
    pub azimuth_offset: f64,
    pub firing_ns: i64,
    pub cos_el: f64,
    pub sin_el: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TimeSpec {
    None,
    DateTime {
        seconds: usize,
        year_base: i64,
        micros: usize,
    },
    UnixSeconds {
        seconds: usize,
        micros: usize,
    },
    TopOfHour {
        micros: usize,
    },
}

/// A validated profile.
#[derive(Debug)]
pub struct Model {
    pub profile: &'static Profile,
    pub(crate) packet_size: usize,
    /// Shortest packet that holds every field decoding reads.
    pub(crate) min_len: usize,
    pub(crate) constants: Vec<(usize, &'static [u8], bool)>,
    pub(crate) blocks_offset: usize,
    pub(crate) block_count: usize,
    pub(crate) block_size: usize,
    pub(crate) flags: Vec<u16>,
    pub(crate) azimuth_offset: usize,
    pub(crate) fine_azimuth_offset: Option<usize>,
    pub(crate) azimuth_scale: f64,
    pub(crate) units_offset: usize,
    pub(crate) unit_size: usize,
    pub(crate) banks: usize,
    pub(crate) firings_per_block: usize,
    /// Units of one firing in one bank.
    pub(crate) units_per_firing: usize,
    pub(crate) distance_unit: f64,
    pub(crate) return_mode_offset: Option<usize>,
    pub(crate) motor_speed_offset: Option<usize>,
    pub(crate) sequence_offset: Option<usize>,
    pub(crate) time: TimeSpec,
    pub(crate) channels: Vec<ChannelGeom>,
    pub(crate) period_ns: i64,
    pub(crate) offset_ns: i64,
    pub(crate) from_end: bool,
}

impl Model {
    /// The model of a profile name or alias (`extra` profiles first).
    pub fn find(name: &str, extra: Option<&[Profile]>) -> Result<Self> {
        Self::new(find(name, extra.unwrap_or_default())?)
    }

    /// Validate a profile.
    pub fn new(p: &'static Profile) -> Result<Self> {
        let bad = |what: String| Error::Config(format!("LiDAR profile {:?}: {what}", p.name));
        let l = p.layout.as_ref().ok_or_else(|| bad("no layout".into()))?;
        let n = p.channels.len();
        let us = |x: u32| x as usize;
        let (banks, fpb) = (
            us(l.blocks_per_firing.max(1)),
            us(l.firings_per_block.max(1)),
        );
        let upb = us(l.units_per_block);
        if n == 0 {
            return Err(bad("no channels".into()));
        }
        if upb % fpb != 0 || banks * (upb / fpb) != n {
            return Err(bad(format!(
                "{banks} bank(s) x {upb} units / {fpb} firing(s) per block do not make {n} channels"
            )));
        }
        let block_count = us(l.block_count);
        if block_count == 0 || block_count % banks != 0 {
            return Err(bad(format!(
                "{block_count} blocks are not whole firings of {banks} bank(s)"
            )));
        }
        if l.unit_size < 3 {
            return Err(bad(
                "units hold at least a u16 distance and a u8 intensity".into()
            ));
        }
        let block_size = us(l.block_size);
        let units_end = us(l.units_offset) + upb * us(l.unit_size);
        let flags_end = if l.block_flags.is_empty() { 0 } else { 2 };
        let az_end = us(l.azimuth_offset) + 2;
        let fine_end = l.fine_azimuth_offset.map_or(0, |o| us(o) + 1);
        if units_end.max(az_end).max(fine_end).max(flags_end) > block_size {
            return Err(bad(format!(
                "block fields exceed the {block_size}-byte block"
            )));
        }
        if l.block_flags.iter().any(|f| *f > 0xFFFF) {
            return Err(bad("block flags are u16".into()));
        }
        let blocks_end = us(l.blocks_offset) + block_count * block_size;
        let size = us(l.packet_size);
        let mut ends = vec![("blocks", blocks_end)];
        let mut field = |name: &'static str, o: Option<u32>, len: usize| {
            if let Some(o) = o {
                ends.push((name, us(o) + len));
            }
        };
        field("return_mode_offset", l.return_mode_offset, 1);
        field("motor_speed_offset", l.motor_speed_offset, 2);
        field("sequence_offset", l.sequence_offset, 4);
        let ts = l.timestamp.unwrap_or_default();
        let time =
            match Kind::try_from(ts.kind).map_err(|_| bad("unknown timestamp kind".into()))? {
                Kind::None => TimeSpec::None,
                Kind::DateTime => {
                    field("timestamp.seconds_offset", Some(ts.seconds_offset), 6);
                    field("timestamp.micros_offset", Some(ts.micros_offset), 4);
                    TimeSpec::DateTime {
                        seconds: us(ts.seconds_offset),
                        year_base: ts.year_base.into(),
                        micros: us(ts.micros_offset),
                    }
                }
                Kind::UnixSeconds => {
                    field("timestamp.seconds_offset", Some(ts.seconds_offset), 6);
                    field("timestamp.micros_offset", Some(ts.micros_offset), 4);
                    TimeSpec::UnixSeconds {
                        seconds: us(ts.seconds_offset),
                        micros: us(ts.micros_offset),
                    }
                }
                Kind::TopOfHour => {
                    field("timestamp.micros_offset", Some(ts.micros_offset), 4);
                    TimeSpec::TopOfHour {
                        micros: us(ts.micros_offset),
                    }
                }
            };
        let mut constants = Vec::new();
        for c in &l.constants {
            let end = us(c.offset) + c.value.len();
            if c.verify {
                ends.push(("a verified constant", end));
            } else if end > size {
                return Err(bad(format!("a constant ends at {end}, after the packet")));
            }
            constants.push((us(c.offset), c.value.as_slice(), c.verify));
        }
        if let Some((name, end)) = ends.iter().find(|(_, end)| *end > size) {
            return Err(bad(format!(
                "{name} ends at {end}, after the {size}-byte packet"
            )));
        }
        let min_len = ends.iter().map(|(_, e)| *e).max().unwrap_or(size);
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(l.azimuth_scale) || !positive(l.distance_unit) {
            return Err(bad(
                "azimuth_scale and distance_unit must be positive".into()
            ));
        }
        if l.fine_azimuth_offset.is_some() && l.azimuth_scale * 360.0 > f64::from(u32::MAX) {
            return Err(bad("azimuth_scale is too large".into()));
        }
        if l.fine_azimuth_offset.is_none() && l.azimuth_scale * 360.0 > 65536.0 {
            return Err(bad(
                "a u16 azimuth cannot hold 360 degrees at azimuth_scale".into(),
            ));
        }
        for m in &p.return_modes {
            let r = m.returns.len();
            if m.code > 0xFF || r == 0 {
                return Err(bad(format!(
                    "return mode {:?} needs a u8 code and at least one return",
                    m.name
                )));
            }
        }
        if p.return_modes.is_empty() {
            return Err(bad("no return modes".into()));
        }
        let timing = p.timing.unwrap_or_default();
        if timing.firing_period_ns <= 0 {
            return Err(bad("timing.firing_period_ns must be positive".into()));
        }
        if !(p.min_range >= 0.0 && p.max_range > p.min_range) {
            return Err(bad("needs 0 <= min_range < max_range".into()));
        }
        let channels = p
            .channels
            .iter()
            .map(|c| {
                let el = c.elevation.to_radians();
                ChannelGeom {
                    elevation: c.elevation,
                    azimuth_offset: c.azimuth_offset,
                    firing_ns: c.firing_offset_ns.into(),
                    cos_el: el.cos(),
                    sin_el: el.sin(),
                }
            })
            .collect();
        Ok(Self {
            profile: p,
            packet_size: size,
            min_len,
            constants,
            blocks_offset: us(l.blocks_offset),
            block_count,
            block_size,
            flags: l.block_flags.iter().map(|f| *f as u16).collect(),
            azimuth_offset: us(l.azimuth_offset),
            fine_azimuth_offset: l.fine_azimuth_offset.map(us),
            azimuth_scale: l.azimuth_scale,
            units_offset: us(l.units_offset),
            unit_size: us(l.unit_size),
            banks,
            firings_per_block: fpb,
            units_per_firing: upb / fpb,
            distance_unit: l.distance_unit,
            return_mode_offset: l.return_mode_offset.map(us),
            motor_speed_offset: l.motor_speed_offset.map(us),
            sequence_offset: l.sequence_offset.map(us),
            time,
            channels,
            period_ns: timing.firing_period_ns,
            offset_ns: timing.offset_ns,
            from_end: timing.from_end,
        })
    }

    pub fn name(&self) -> &str {
        &self.profile.name
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Size of the packets [`Model::encode`] writes.
    pub fn packet_size(&self) -> usize {
        self.packet_size
    }
}

#[cfg(test)]
mod tests;
