//! Raw <-> physical conversions that one factor and offset plus exact value
//! table entries cannot express: FIBEX compu methods with texttable ranges,
//! piecewise linear scales and rational functions.
//!
//! A coding is a list of scales, each a raw interval with a conversion: a
//! name (texttable) or a rational function `N(x) / D(x)` of the raw value.
//! Decoding picks the first name scale whose interval holds the raw value
//! (when names are wanted), else the first rational scale. Encoding takes a
//! name to its scale's lower limit, and a number to the raw value of the
//! first rational scale that can be inverted at it: exactly for a linear
//! function, by bisection over a scale on which the function is strictly
//! monotonic, and not at all otherwise.

use std::ops::Bound;

/// How the raw values of a scale convert.
#[derive(Clone, Debug, PartialEq)]
pub enum Conversion {
    /// A texttable name.
    Text(String),
    /// `N(x) / D(x)`: coefficients from the constant term up.
    Rational {
        numerator: Vec<f64>,
        denominator: Vec<f64>,
    },
}

/// A raw interval and its conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct Scale {
    pub lower: Bound<f64>,
    pub upper: Bound<f64>,
    pub conversion: Conversion,
}

impl Scale {
    fn contains(&self, raw: f64) -> bool {
        let above = match self.lower {
            Bound::Included(l) => raw >= l,
            Bound::Excluded(l) => raw > l,
            Bound::Unbounded => true,
        };
        let below = match self.upper {
            Bound::Included(u) => raw <= u,
            Bound::Excluded(u) => raw < u,
            Bound::Unbounded => true,
        };
        above && below
    }

    /// The raw value a name encodes to: the lower limit (the upper one for
    /// a scale open below).
    fn representative(&self) -> Option<f64> {
        match (self.lower, self.upper) {
            (Bound::Included(l), _) => Some(l),
            (Bound::Excluded(l), _) => Some(l.floor() + 1.0),
            (Bound::Unbounded, Bound::Included(u)) => Some(u),
            (Bound::Unbounded, Bound::Excluded(u)) => Some(u.ceil() - 1.0),
            (Bound::Unbounded, Bound::Unbounded) => None,
        }
    }
}

/// How a rational scale is inverted over its raw domain.
#[derive(Clone, Debug, PartialEq)]
enum Inverse {
    /// `a + b x`, `b != 0`.
    Linear { a: f64, b: f64 },
    /// The same physical value for every raw value.
    Constant(f64),
    /// Strictly monotonic on `lo..=hi`.
    Monotonic { lo: f64, hi: f64 },
    /// Not invertible (or no raw value of the signal lies in the scale).
    None,
}

/// The coding of one signal.
#[derive(Clone, Debug, PartialEq)]
pub struct Coding {
    pub scales: Vec<Scale>,
    /// Per scale; `Inverse::None` for texttable scales.
    inverses: Vec<Inverse>,
    integer: bool,
}

/// Evaluate a polynomial (coefficients from the constant term up).
fn poly(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

/// Integer points examined for monotonicity: every raw value up to this many.
const SAMPLES: u64 = 1 << 16;

impl Coding {
    /// A coding for a signal whose raw values are `raw_min..=raw_max`
    /// (integers when `integer`).
    pub fn new(scales: Vec<Scale>, raw_min: f64, raw_max: f64, integer: bool) -> Self {
        let inverses = scales
            .iter()
            .map(|s| inverse(s, raw_min, raw_max, integer))
            .collect();
        Self {
            scales,
            inverses,
            integer,
        }
    }

    /// The name of a raw value, if a texttable scale holds it.
    pub fn name(&self, raw: f64) -> Option<&str> {
        self.scales.iter().find_map(|s| match &s.conversion {
            Conversion::Text(name) if s.contains(raw) => Some(name.as_str()),
            _ => None,
        })
    }

    /// The physical value of a raw value, if a rational scale holds it.
    pub fn physical(&self, raw: f64) -> Option<f64> {
        self.scales.iter().find_map(|s| match &s.conversion {
            Conversion::Rational {
                numerator,
                denominator,
            } if s.contains(raw) => Some(poly(numerator, raw) / poly(denominator, raw)),
            _ => None,
        })
    }

    /// The raw value of a name.
    pub fn raw_of_name(&self, name: &str) -> Option<f64> {
        self.scales.iter().find_map(|s| match &s.conversion {
            Conversion::Text(n) if n == name => s.representative(),
            _ => None,
        })
    }

    /// The names of the texttable scales.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.scales.iter().filter_map(|s| match &s.conversion {
            Conversion::Text(n) => Some(n.as_str()),
            Conversion::Rational { .. } => None,
        })
    }

    /// The raw value of a physical value; `Err` says why there is none.
    pub fn raw_of_physical(&self, phys: f64) -> std::result::Result<f64, String> {
        let mut not_invertible = false;
        for (scale, inverse) in self.scales.iter().zip(&self.inverses) {
            let Conversion::Rational {
                numerator,
                denominator,
            } = &scale.conversion
            else {
                continue;
            };
            let f = |x: f64| poly(numerator, x) / poly(denominator, x);
            let found = match *inverse {
                Inverse::Linear { a, b } => {
                    let x = (phys - a) / b;
                    let x = if self.integer { x.round() } else { x };
                    Some(x).filter(|&x| scale.contains(x))
                }
                Inverse::Constant(c) => scale
                    .representative()
                    .filter(|_| (phys - c).abs() <= 1e-9 * c.abs().max(1.0)),
                Inverse::Monotonic { lo, hi } => self.bisect(f, lo, hi, phys),
                Inverse::None => {
                    not_invertible = true;
                    None
                }
            };
            if let Some(x) = found {
                return Ok(x);
            }
        }
        Err(if not_invertible {
            format!(
                "{phys} is outside the invertible scales of its coding (a scale whose function \
                 is not monotonic over its raw values cannot be encoded)"
            )
        } else {
            format!("{phys} is outside every scale of its coding")
        })
    }

    /// The raw value in `lo..=hi` whose physical value is nearest `phys`,
    /// if `phys` lies within the scale's physical range (to half a raw step
    /// at its ends for integers).
    fn bisect(&self, f: impl Fn(f64) -> f64, lo: f64, hi: f64, phys: f64) -> Option<f64> {
        let increasing = f(hi) > f(lo);
        // g increases with x.
        let g = |x: f64| if increasing { f(x) } else { -f(x) };
        let target = if increasing { phys } else { -phys };
        let step = if self.integer { 1.0 } else { 0.0 };
        let slack = |a: f64, b: f64| (g(b) - g(a)).abs() / 2.0 + 1e-9 * target.abs().max(1.0);
        let (glo, ghi) = (g(lo), g(hi));
        if target < glo - slack(lo, (lo + step).min(hi))
            || target > ghi + slack((hi - step).max(lo), hi)
        {
            return None;
        }
        if target <= glo {
            return Some(lo);
        }
        if target >= ghi {
            return Some(hi);
        }
        // Invariant: g(a) < target <= g(b).
        let (mut a, mut b) = (lo, hi);
        if self.integer {
            while b - a > 1.0 {
                let m = (a + (b - a) / 2.0).floor();
                if g(m) < target {
                    a = m;
                } else {
                    b = m;
                }
            }
            Some(if target - g(a) <= g(b) - target { a } else { b })
        } else {
            for _ in 0..200 {
                let m = a + (b - a) / 2.0;
                if m <= a || m >= b {
                    break;
                }
                if g(m) < target {
                    a = m;
                } else {
                    b = m;
                }
            }
            Some(b)
        }
    }
}

/// How a scale inverts over the raw values `raw_min..=raw_max`.
fn inverse(scale: &Scale, raw_min: f64, raw_max: f64, integer: bool) -> Inverse {
    let Conversion::Rational {
        numerator,
        denominator,
    } = &scale.conversion
    else {
        return Inverse::None;
    };
    let trim = |c: &[f64]| {
        let n = c.iter().rposition(|v| *v != 0.0).map_or(0, |i| i + 1);
        c[..n].to_vec()
    };
    let (n, d) = (trim(numerator), trim(denominator));
    if let ([d0], [] | [_] | [_, _]) = (d.as_slice(), n.as_slice()) {
        let a = n.first().copied().unwrap_or(0.0) / d0;
        let b = n.get(1).copied().unwrap_or(0.0) / d0;
        if !a.is_finite() || !b.is_finite() {
            return Inverse::None;
        }
        return if b == 0.0 {
            Inverse::Constant(a)
        } else {
            Inverse::Linear { a, b }
        };
    }
    // The scale's raw domain, within the signal's raw values.
    let mut lo = match scale.lower {
        Bound::Included(l) => l.max(raw_min),
        Bound::Excluded(l) if integer => (l.floor() + 1.0).max(raw_min),
        Bound::Excluded(l) => l.max(raw_min),
        Bound::Unbounded => raw_min,
    };
    let mut hi = match scale.upper {
        Bound::Included(u) => u.min(raw_max),
        Bound::Excluded(u) if integer => (u.ceil() - 1.0).min(raw_max),
        Bound::Excluded(u) => u.min(raw_max),
        Bound::Unbounded => raw_max,
    };
    if integer {
        (lo, hi) = (lo.ceil(), hi.floor());
    }
    if !(lo.is_finite() && hi.is_finite()) || lo > hi {
        return Inverse::None;
    }
    if lo == hi {
        return if f64::is_finite(poly(&n, lo) / poly(&d, lo)) {
            Inverse::Monotonic { lo, hi }
        } else {
            Inverse::None
        };
    }
    // Strictly monotonic at every integer raw value (or at evenly spaced
    // points, for a wide or real domain).
    let span = hi - lo;
    let points = if integer && span < SAMPLES as f64 {
        span as u64
    } else {
        SAMPLES
    };
    let f = |x: f64| poly(&n, x) / poly(&d, x);
    let mut previous = f(lo);
    let mut direction = 0.0;
    for i in 1..=points {
        let x = if i == points {
            hi
        } else {
            lo + span * i as f64 / points as f64
        };
        let x = if integer { x.round() } else { x };
        let y = f(x);
        let step = y - previous;
        if !y.is_finite() || step == 0.0 || step.signum() * direction < 0.0 {
            return Inverse::None;
        }
        direction = step.signum();
        previous = y;
    }
    // A denominator root between sample points would break monotonicity;
    // with a real domain, require the denominator not to change sign.
    if !integer || points < span as u64 {
        let d_lo = poly(&d, lo);
        if (0..=points).any(|i| poly(&d, lo + span * i as f64 / points as f64) * d_lo <= 0.0) {
            return Inverse::None;
        }
    }
    Inverse::Monotonic { lo, hi }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rational(lower: f64, upper: f64, n: &[f64], d: &[f64]) -> Scale {
        Scale {
            lower: Bound::Included(lower),
            upper: Bound::Included(upper),
            conversion: Conversion::Rational {
                numerator: n.to_vec(),
                denominator: d.to_vec(),
            },
        }
    }

    fn text(lower: f64, upper: f64, name: &str) -> Scale {
        Scale {
            lower: Bound::Included(lower),
            upper: Bound::Included(upper),
            conversion: Conversion::Text(name.into()),
        }
    }

    #[test]
    fn texttable_ranges() {
        let c = Coding::new(
            vec![text(0.0, 3.0, "Low"), text(4.0, 15.0, "High")],
            0.0,
            15.0,
            true,
        );
        assert_eq!(c.name(0.0), Some("Low"));
        assert_eq!(c.name(3.0), Some("Low"));
        assert_eq!(c.name(9.0), Some("High"));
        assert_eq!(c.name(16.0), None);
        assert_eq!(c.raw_of_name("High"), Some(4.0));
        assert_eq!(c.raw_of_name("Nope"), None);
        assert!(c.raw_of_physical(1.0).is_err());
    }

    #[test]
    fn piecewise_linear() {
        // 0.5 x for 0..=99, 2 x - 150 for 100..=255.
        let c = Coding::new(
            vec![
                rational(0.0, 99.0, &[0.0, 0.5], &[1.0]),
                rational(100.0, 255.0, &[-150.0, 2.0], &[1.0]),
            ],
            0.0,
            255.0,
            true,
        );
        assert_eq!(c.physical(10.0), Some(5.0));
        assert_eq!(c.physical(100.0), Some(50.0));
        assert_eq!(c.physical(255.0), Some(360.0));
        assert_eq!(c.raw_of_physical(5.0), Ok(10.0));
        assert_eq!(c.raw_of_physical(49.5), Ok(99.0));
        assert_eq!(c.raw_of_physical(52.0), Ok(101.0));
        assert_eq!(c.raw_of_physical(360.0), Ok(255.0));
        assert!(c.raw_of_physical(400.0).is_err());
        assert!(c.raw_of_physical(-1.0).is_err());
    }

    #[test]
    fn rational_functions() {
        // 1000 / (x + 1): decreasing over 0..=255.
        let c = Coding::new(
            vec![rational(0.0, 255.0, &[1000.0], &[1.0, 1.0])],
            0.0,
            255.0,
            true,
        );
        assert_eq!(c.physical(0.0), Some(1000.0));
        assert_eq!(c.physical(9.0), Some(100.0));
        for raw in [0.0, 1.0, 9.0, 99.0, 254.0, 255.0] {
            let phys = c.physical(raw).unwrap();
            assert_eq!(c.raw_of_physical(phys), Ok(raw), "{raw}");
        }
        // Rounded to the nearest raw value.
        assert_eq!(c.raw_of_physical(101.0), Ok(9.0));
        assert!(c.raw_of_physical(2000.0).is_err());
        // A real domain: x^2 / 4 over 0..=10.
        let r = Coding::new(
            vec![rational(0.0, 10.0, &[0.0, 0.0, 1.0], &[4.0])],
            f64::MIN,
            f64::MAX,
            false,
        );
        let x = r.raw_of_physical(2.25).unwrap();
        assert!((x - 3.0).abs() < 1e-9, "{x}");
        // (x - 8)^2 is not monotonic over 0..=15: decode only.
        let p = Coding::new(
            vec![rational(0.0, 15.0, &[64.0, -16.0, 1.0], &[1.0])],
            0.0,
            15.0,
            true,
        );
        assert_eq!(p.physical(10.0), Some(4.0));
        assert!(p
            .raw_of_physical(4.0)
            .unwrap_err()
            .contains("not monotonic"));
        // Without limits, a non-linear function over an unbounded raw domain.
        let unbounded = Coding::new(
            vec![Scale {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
                conversion: Conversion::Rational {
                    numerator: vec![0.0, 0.0, 1.0],
                    denominator: vec![1.0],
                },
            }],
            f64::NEG_INFINITY,
            f64::INFINITY,
            false,
        );
        assert!(unbounded.raw_of_physical(4.0).is_err());
    }

    #[test]
    fn open_limits_and_single_values() {
        // x^2 on the single raw value 3, then 10 + x above 3 (open).
        let c = Coding::new(
            vec![
                rational(3.0, 3.0, &[0.0, 0.0, 1.0], &[1.0]),
                Scale {
                    lower: Bound::Excluded(3.0),
                    upper: Bound::Unbounded,
                    conversion: Conversion::Rational {
                        numerator: vec![10.0, 1.0],
                        denominator: vec![1.0],
                    },
                },
            ],
            0.0,
            255.0,
            true,
        );
        assert_eq!(c.physical(3.0), Some(9.0));
        assert_eq!(c.physical(4.0), Some(14.0));
        assert_eq!(c.physical(2.0), None);
        assert_eq!(c.raw_of_physical(9.0), Ok(3.0));
        assert_eq!(c.raw_of_physical(14.0), Ok(4.0));
        // 13 would be raw 3, which the open scale leaves out.
        assert!(c.raw_of_physical(13.0).is_err());
    }
}
