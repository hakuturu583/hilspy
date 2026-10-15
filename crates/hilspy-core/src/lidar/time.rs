//! Calendar conversions for packet timestamps (UTC, proleptic Gregorian).

/// Days since 1970-01-01 of a civil date (Howard Hinnant's algorithm).
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date of days since 1970-01-01.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Unix seconds of a UTC date and time; None if a field is out of range.
pub(crate) fn unix_seconds(y: i64, mo: u8, d: u8, h: u8, mi: u8, s: u8) -> Option<i64> {
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    let days = days_from_civil(y, mo.into(), d.into());
    Some(days * 86_400 + i64::from(h) * 3600 + i64::from(mi) * 60 + i64::from(s))
}

/// `(year, month, day, hour, minute, second)` of Unix seconds.
pub(crate) fn utc(secs: i64) -> (i64, u8, u8, u8, u8, u8) {
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    (
        y,
        m as u8,
        d as u8,
        (s / 3600) as u8,
        (s / 60 % 60) as u8,
        (s % 60) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        assert_eq!(unix_seconds(1970, 1, 1, 0, 0, 0), Some(0));
        assert_eq!(unix_seconds(2024, 2, 29, 12, 34, 56), Some(1_709_210_096));
        assert_eq!(utc(1_709_210_096), (2024, 2, 29, 12, 34, 56));
        for s in [-86_401, 0, 951_782_400, 4_102_444_799] {
            let (y, mo, d, h, mi, sec) = utc(s);
            assert_eq!(unix_seconds(y, mo, d, h, mi, sec), Some(s));
        }
        assert_eq!(unix_seconds(2024, 13, 1, 0, 0, 0), None);
    }
}
