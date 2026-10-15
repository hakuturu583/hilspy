//! Body fields of the blocks `libsbf` parses, as name/value pairs.

use libsbf::Messages;

/// A field value. `Null` is SBF's do-not-use value.
#[derive(Clone, Debug, PartialEq)]
pub enum Field {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A group of fields (an optional sub-block).
    Map(Vec<(String, Field)>),
}

pub(crate) trait IntoField {
    fn into_field(self) -> Field;
}

macro_rules! int_field {
    ($($t:ty),*) => {$(
        impl IntoField for $t {
            fn into_field(self) -> Field {
                Field::Int(i64::from(self))
            }
        }
    )*};
}
int_field!(u8, u16, u32, i8, i16, i32);

impl IntoField for f32 {
    fn into_field(self) -> Field {
        // Through the shortest decimal form, so 0.1f32 reads as 0.1.
        Field::Float(self.to_string().parse().unwrap_or(f64::from(self)))
    }
}

impl IntoField for f64 {
    fn into_field(self) -> Field {
        Field::Float(self)
    }
}

impl IntoField for bool {
    fn into_field(self) -> Field {
        Field::Bool(self)
    }
}

impl IntoField for String {
    fn into_field(self) -> Field {
        Field::Str(self)
    }
}

impl<T: IntoField> IntoField for Option<T> {
    fn into_field(self) -> Field {
        self.map_or(Field::Null, IntoField::into_field)
    }
}

impl IntoField for Field {
    fn into_field(self) -> Field {
        self
    }
}

macro_rules! fields {
    ($($name:literal => $value:expr),* $(,)?) => {
        vec![$(($name.to_string(), IntoField::into_field($value))),*]
    };
}

fn debug<T: std::fmt::Debug>(v: T) -> String {
    format!("{v:?}")
}

/// Map an optional sub-block to a group of fields.
fn group<T>(v: &Option<T>, f: impl FnOnce(&T) -> Vec<(String, Field)>) -> Field {
    v.as_ref().map_or(Field::Null, |v| Field::Map(f(v)))
}

pub(crate) fn fields(message: &Messages) -> Option<Vec<(String, Field)>> {
    Some(match message {
        Messages::PVTGeodetic(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "mode" => u8::from(m.pvt_mode()),
            "mode_name" => m.pvt_mode().to_string(),
            "mode_2d" => m.mode_flags().contains(libsbf::PvtModeFlags::MODE_2D),
            "error" => u8::from(m.error),
            "latitude" => m.latitude,
            "longitude" => m.longitude,
            "height" => m.height,
            "undulation" => m.undulation,
            "vn" => m.vn,
            "ve" => m.ve,
            "vu" => m.vu,
            "cog" => m.cog,
            "rx_clk_bias" => m.rx_clk_bias,
            "rx_clk_drift" => m.rx_clk_drift,
            "time_system" => m.time_system,
            "datum" => m.datum.map(u8::from),
            "nr_sv" => m.nr_sv,
            "reference_id" => m.reference_id,
            "mean_corr_age" => m.mean_corr_age,
            "signal_info" => m.signal_info,
            "nr_bases" => m.nr_bases,
            "latency" => m.latency,
            "h_accuracy" => m.h_accuracy,
            "v_accuracy" => m.v_accuracy,
        ],
        Messages::PVTCartesian(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "mode" => u8::from(m.pvt_mode()),
            "mode_name" => m.pvt_mode().to_string(),
            "error" => u8::from(m.error),
            "x" => m.x,
            "y" => m.y,
            "z" => m.z,
            "undulation" => m.undulation,
            "vx" => m.vx,
            "vy" => m.vy,
            "vz" => m.vz,
            "cog" => m.cog,
            "rx_clk_bias" => m.rx_clk_bias,
            "rx_clk_drift" => m.rx_clk_drift,
            "time_system" => m.time_system,
            "datum" => m.datum.map(u8::from),
            "nr_sv" => m.nr_sv,
            "reference_id" => m.reference_id,
            "mean_corr_age" => m.mean_corr_age,
            "signal_info" => m.signal_info,
            "nr_bases" => m.nr_bases,
            "latency" => m.latency,
            "h_accuracy" => m.h_accuracy,
            "v_accuracy" => m.v_accuracy,
        ],
        Messages::AttEuler(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "nr_sv" => m.nrsv,
            "mode" => debug(m.mode()),
            "heading" => m.heading,
            "pitch" => m.pitch,
            "roll" => m.roll,
            "pitch_dot" => m.pitch_dot,
            "roll_dot" => m.roll_dot,
            "heading_dot" => m.heading_dot,
        ],
        Messages::INSNavGeod(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "error" => m.error,
            "info" => m.info,
            "gnss_mode" => u8::from(m.pvt_mode()),
            "gnss_age" => m.gnss_age,
            "latitude" => m.latitude,
            "longitude" => m.longitude,
            "height" => m.height,
            "undulation" => m.undulation,
            "accuracy" => m.accuracy,
            "latency" => m.latency,
            "datum" => m.datum.map(u8::from),
            "sb_list" => m.sb_list,
            "pos_std_dev" => group(&m.pos_std_dev, |s| fields![
                "latitude" => s.latitude_std_dev,
                "longitude" => s.longitude_std_dev,
                "height" => s.height_std_dev,
            ]),
            "att" => group(&m.att, |a| fields![
                "heading" => a.heading,
                "pitch" => a.pitch,
                "roll" => a.roll,
            ]),
            "att_std_dev" => group(&m.att_std_dev, |a| fields![
                "heading" => a.heading_std_dev,
                "pitch" => a.pitch_std_dev,
                "roll" => a.roll_std_dev,
            ]),
            "vel" => group(&m.vel, |v| fields![
                "ve" => v.ve,
                "vn" => v.vn,
                "vu" => v.vu,
            ]),
            "vel_std_dev" => group(&m.vel_std_dev, |v| fields![
                "ve" => v.ve_std_dev,
                "vn" => v.vn_std_dev,
                "vu" => v.vu_std_dev,
            ]),
        ],
        Messages::ReceiverTime(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "utc_year" => m.utc_year,
            "utc_month" => m.utc_month,
            "utc_day" => m.utc_day,
            "utc_hour" => m.utc_hour,
            "utc_min" => m.utc_min,
            "utc_sec" => m.utc_sec,
            "delta_ls" => m.delta_ls,
            "sync_level" => m.sync_level,
        ],
        Messages::DOP(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "nr_sv" => m.nr_sv,
            "pdop" => m.pdop_value(),
            "tdop" => m.tdop_value(),
            "hdop" => m.hdop_value(),
            "vdop" => m.vdop_value(),
            "hpl" => m.hpl,
            "vpl" => m.vpl,
        ],
        Messages::PosCovGeodetic(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "mode" => u8::from(m.pvt_mode()),
            "error" => u8::from(m.error()),
            "cov_latlat" => m.cov_latlat,
            "cov_lonlon" => m.cov_lonlon,
            "cov_hgthgt" => m.cov_hgthgt,
            "cov_bb" => m.cov_bb,
            "cov_latlon" => m.cov_latlon,
            "cov_lathgt" => m.cov_lathgt,
            "cov_latb" => m.cov_latb,
            "cov_lonhgt" => m.cov_lonhgt,
            "cov_lonb" => m.cov_lonb,
            "cov_hb" => m.cov_hb,
        ],
        Messages::VelCovGeodetic(m) => fields![
            "tow" => m.tow,
            "wnc" => m.wnc,
            "mode" => u8::from(m.pvt_mode()),
            "error" => u8::from(m.error()),
            "cov_vnvn" => m.cov_vnvn,
            "cov_veve" => m.cov_veve,
            "cov_vuvu" => m.cov_vuvu,
            "cov_dd" => m.cov_dd,
            "cov_vnve" => m.cov_vnve,
            "cov_vnvu" => m.cov_vnvu,
            "cov_vnd" => m.cov_vnd,
            "cov_vevu" => m.cov_vevu,
            "cov_ved" => m.cov_ved,
            "cov_vud" => m.cov_vud,
        ],
        Messages::EndOfPVT(m) => fields!["tow" => m.tow, "wnc" => m.wnc],
        Messages::EndOfAtt(m) => fields!["tow" => m.tow, "wnc" => m.wnc],
        Messages::EndOfMeas(m) => fields!["tow" => m.tow, "wnc" => m.wnc],
        _ => return None,
    })
}
