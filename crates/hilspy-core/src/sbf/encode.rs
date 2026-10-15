//! Encoders for the blocks a simulated GNSS/INS receiver sends. `None` fields
//! are written as SBF's do-not-use values.

use super::frame_block;
use crate::error::Result;

const DNU_F4: f32 = -2e10;
const DNU_F8: f64 = -2e10;

fn f4(v: Option<f64>) -> [u8; 4] {
    v.map_or(DNU_F4, |v| v as f32).to_le_bytes()
}

fn f8(v: Option<f64>) -> [u8; 8] {
    v.unwrap_or(DNU_F8).to_le_bytes()
}

fn u1(v: Option<u8>) -> u8 {
    v.unwrap_or(u8::MAX)
}

fn u2(v: Option<u16>) -> [u8; 2] {
    v.unwrap_or(u16::MAX).to_le_bytes()
}

fn u4(v: Option<u32>) -> [u8; 4] {
    v.unwrap_or(u32::MAX).to_le_bytes()
}

/// PVTGeodetic (block 4007, revision 2): position in WGS84 radians/metres,
/// velocity in m/s (north, east, up).
#[derive(Clone, Debug, PartialEq)]
pub struct PvtGeodetic {
    pub tow: Option<u32>,
    pub wnc: Option<u16>,
    /// PVT mode (bits 0-3): 0 no fix, 1 stand-alone, 2 DGNSS, 4 RTK fixed,
    /// 5 RTK float, ...; bit 7 = 2D.
    pub mode: u8,
    /// 0 = no error.
    pub error: u8,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub height: Option<f64>,
    pub undulation: Option<f64>,
    pub vn: Option<f64>,
    pub ve: Option<f64>,
    pub vu: Option<f64>,
    /// Course over ground, degrees.
    pub cog: Option<f64>,
    pub rx_clk_bias: Option<f64>,
    pub rx_clk_drift: Option<f64>,
    /// 0 = GPS time.
    pub time_system: Option<u8>,
    /// 0 = WGS84.
    pub datum: Option<u8>,
    pub nr_sv: Option<u8>,
    pub wa_corr_info: u8,
    pub reference_id: Option<u16>,
    pub mean_corr_age: Option<u16>,
    pub signal_info: u32,
    pub alert_flag: u8,
    pub nr_bases: u8,
    pub ppp_info: u16,
    pub latency: Option<u16>,
    /// Horizontal accuracy, centimetres.
    pub h_accuracy: Option<u16>,
    /// Vertical accuracy, centimetres.
    pub v_accuracy: Option<u16>,
    pub misc: u8,
}

impl Default for PvtGeodetic {
    fn default() -> Self {
        Self {
            tow: None,
            wnc: None,
            mode: 0,
            error: 0,
            latitude: None,
            longitude: None,
            height: None,
            undulation: None,
            vn: None,
            ve: None,
            vu: None,
            cog: None,
            rx_clk_bias: None,
            rx_clk_drift: None,
            time_system: Some(0),
            datum: Some(0),
            nr_sv: None,
            wa_corr_info: 0,
            reference_id: None,
            mean_corr_age: None,
            signal_info: 0,
            alert_flag: 0,
            nr_bases: 0,
            ppp_info: 0,
            latency: None,
            h_accuracy: None,
            v_accuracy: None,
            misc: 0,
        }
    }
}

impl PvtGeodetic {
    pub const NUMBER: u16 = 4007;
    pub const REVISION: u8 = 2;

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut b = Vec::with_capacity(88);
        b.extend_from_slice(&u4(self.tow));
        b.extend_from_slice(&u2(self.wnc));
        b.push(self.mode);
        b.push(self.error);
        b.extend_from_slice(&f8(self.latitude));
        b.extend_from_slice(&f8(self.longitude));
        b.extend_from_slice(&f8(self.height));
        b.extend_from_slice(&f4(self.undulation));
        b.extend_from_slice(&f4(self.vn));
        b.extend_from_slice(&f4(self.ve));
        b.extend_from_slice(&f4(self.vu));
        b.extend_from_slice(&f4(self.cog));
        b.extend_from_slice(&f8(self.rx_clk_bias));
        b.extend_from_slice(&f4(self.rx_clk_drift));
        b.push(u1(self.time_system));
        b.push(u1(self.datum));
        b.push(u1(self.nr_sv));
        b.push(self.wa_corr_info);
        b.extend_from_slice(&u2(self.reference_id));
        b.extend_from_slice(&u2(self.mean_corr_age));
        b.extend_from_slice(&self.signal_info.to_le_bytes());
        b.push(self.alert_flag);
        b.push(self.nr_bases);
        b.extend_from_slice(&self.ppp_info.to_le_bytes());
        b.extend_from_slice(&u2(self.latency));
        b.extend_from_slice(&u2(self.h_accuracy));
        b.extend_from_slice(&u2(self.v_accuracy));
        b.push(self.misc);
        frame_block(Self::NUMBER, Self::REVISION, &b)
    }
}

/// AttEuler (block 5938, revision 1): attitude in degrees, rates in deg/s.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttEuler {
    pub tow: Option<u32>,
    pub wnc: Option<u16>,
    pub nr_sv: Option<u8>,
    pub error: u8,
    /// 0 no attitude, 1/2 heading-pitch float/fixed, 3/4 heading-pitch-roll
    /// float/fixed.
    pub mode: u16,
    pub heading: Option<f64>,
    pub pitch: Option<f64>,
    pub roll: Option<f64>,
    pub pitch_dot: Option<f64>,
    pub roll_dot: Option<f64>,
    pub heading_dot: Option<f64>,
}

impl AttEuler {
    pub const NUMBER: u16 = 5938;
    pub const REVISION: u8 = 1;

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut b = Vec::with_capacity(36);
        b.extend_from_slice(&u4(self.tow));
        b.extend_from_slice(&u2(self.wnc));
        b.push(u1(self.nr_sv));
        b.push(self.error);
        b.extend_from_slice(&self.mode.to_le_bytes());
        b.extend_from_slice(&[0, 0]);
        for v in [
            self.heading,
            self.pitch,
            self.roll,
            self.pitch_dot,
            self.roll_dot,
            self.heading_dot,
        ] {
            b.extend_from_slice(&f4(v));
        }
        frame_block(Self::NUMBER, Self::REVISION, &b)
    }
}
