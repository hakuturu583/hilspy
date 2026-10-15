//! A virtual clock for synchronous simulation.
//!
//! Channels opened with a [`Clock`] run on its time instead of the wall
//! clock: periodic sends ([`crate::Channel::every`]) and the paced packets
//! of a message (a LiDAR revolution) are scheduled on it, and happen inside
//! [`Clock::advance`], in time order, when the simulation steps past them.
//! Messages are stamped with its time. Nothing is sent between steps, so a
//! simulator that steps the clock with its own time step (CARLA's
//! synchronous mode) drives every simulated device deterministically.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};

/// Work scheduled on the clock; it gets the clock and its due time (ns).
type Task = Box<dyn FnOnce(&Clock, i64) + Send>;

struct Entry {
    at: i64,
    seq: u64,
    task: Task,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        (self.at, self.seq) == (other.at, other.seq)
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    /// Reversed, so the heap pops the earliest (then first scheduled) entry.
    fn cmp(&self, other: &Self) -> Ordering {
        (other.at, other.seq).cmp(&(self.at, self.seq))
    }
}

struct State {
    now: i64,
    seq: u64,
    queue: BinaryHeap<Entry>,
}

struct Inner {
    state: Mutex<State>,
    /// Held for a whole advance, so steps from several threads do not
    /// interleave their sends.
    advancing: Mutex<()>,
}

/// Simulated time in seconds (nanosecond resolution). Cloning shares the
/// clock.
#[derive(Clone)]
pub struct Clock {
    inner: Arc<Inner>,
}

pub(crate) fn to_ns(seconds: f64) -> Result<i64> {
    if !seconds.is_finite() || seconds.abs() > 9.0e9 {
        return Err(Error::Config(format!("{seconds} is not a valid time")));
    }
    Ok((seconds * 1e9).round() as i64)
}

impl Clock {
    /// A clock at `start` seconds (e.g. 0, or a Unix time to stamp messages
    /// with realistic dates).
    pub fn new(start: f64) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    now: to_ns(start)?,
                    seq: 0,
                    queue: BinaryHeap::new(),
                }),
                advancing: Mutex::new(()),
            }),
        })
    }

    /// The current time, seconds.
    pub fn now(&self) -> f64 {
        self.now_ns() as f64 * 1e-9
    }

    pub(crate) fn now_ns(&self) -> i64 {
        self.inner.state.lock().unwrap().now
    }

    /// When the next scheduled work is due, seconds.
    pub fn next_due(&self) -> Option<f64> {
        let s = self.inner.state.lock().unwrap();
        s.queue.peek().map(|e| e.at as f64 * 1e-9)
    }

    /// Advance by `dt` seconds; see [`Clock::advance_to`].
    pub fn advance(&self, dt: f64) -> Result<()> {
        if dt.is_nan() || dt < 0.0 {
            return Err(Error::Config(format!("cannot advance by {dt} s")));
        }
        let _step = self.inner.advancing.lock().unwrap();
        let target = self.now_ns().saturating_add(to_ns(dt)?);
        self.run_until(target);
        Ok(())
    }

    /// Advance to `t` seconds, running everything due up to and including
    /// `t` in time order (work it schedules within the step runs too), with
    /// [`Clock::now`] reading each one's due time. Moving backwards is an
    /// error.
    pub fn advance_to(&self, t: f64) -> Result<()> {
        let _step = self.inner.advancing.lock().unwrap();
        let target = to_ns(t)?;
        if target < self.now_ns() {
            return Err(Error::Config(format!(
                "cannot move the clock back from {} to {t}",
                self.now()
            )));
        }
        self.run_until(target);
        Ok(())
    }

    fn run_until(&self, target: i64) {
        loop {
            let entry = {
                let mut s = self.inner.state.lock().unwrap();
                if s.queue.peek().is_none_or(|e| e.at > target) {
                    s.now = s.now.max(target);
                    return;
                }
                let e = s.queue.pop().expect("peeked");
                s.now = s.now.max(e.at);
                e
            };
            (entry.task)(self, entry.at);
        }
    }

    /// Run `task` at `at` (ns), or at the next step if `at` has passed.
    pub(crate) fn schedule(&self, at: i64, task: Task) {
        let mut s = self.inner.state.lock().unwrap();
        let seq = s.seq;
        s.seq += 1;
        s.queue.push(Entry { at, seq, task });
    }

    /// Scheduled work not yet run.
    pub fn pending(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }
}

impl std::fmt::Debug for Clock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Clock(now={}, pending={})", self.now(), self.pending())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_work_in_time_order_within_a_step() {
        let clock = Clock::new(10.0).unwrap();
        let log = Arc::new(Mutex::new(Vec::new()));
        for (at, name) in [(10.3, "c"), (10.1, "a"), (10.1, "b"), (11.0, "late")] {
            let log = Arc::clone(&log);
            clock.schedule(
                to_ns(at).unwrap(),
                Box::new(move |c: &Clock, due| {
                    assert_eq!(c.now_ns(), due);
                    log.lock().unwrap().push(name);
                    // Work scheduled during a step runs in that step if due.
                    if name == "a" {
                        let log = Arc::clone(&log);
                        c.schedule(
                            due + 50_000_000,
                            Box::new(move |_: &Clock, _| log.lock().unwrap().push("a+50ms")),
                        );
                    }
                }),
            );
        }
        clock.advance(0.5).unwrap();
        assert_eq!(*log.lock().unwrap(), ["a", "b", "a+50ms", "c"]);
        assert!((clock.now() - 10.5).abs() < 1e-9);
        assert_eq!(clock.next_due(), Some(11.0));
        clock.advance_to(11.0).unwrap();
        assert_eq!(log.lock().unwrap().len(), 5);
        assert!(clock.advance_to(10.0).is_err());
        assert!(clock.advance(-1.0).is_err());
        assert!(clock.advance(f64::NAN).is_err());
        assert!(Clock::new(f64::INFINITY).is_err());
    }
}
