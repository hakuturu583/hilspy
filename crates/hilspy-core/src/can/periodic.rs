use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::bus::Bus;
use super::frame::Frame;
use crate::error::{Error, Result};

struct State {
    frame: Frame,
    period: Duration,
    stopped: bool,
    sent: u64,
    last_error: Option<String>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

/// A frame sent on a fixed period from a background thread (the cyclic
/// messages an ECU emits). Deadlines are absolute, so the period does not
/// drift; a deadline missed by more than one period is skipped rather than
/// sent in a burst. Dropping the task stops it.
pub struct PeriodicTask {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl PeriodicTask {
    pub(crate) fn start(
        bus: Bus,
        frame: Frame,
        period: Duration,
        duration: Option<Duration>,
    ) -> Result<Self> {
        if period.is_zero() {
            return Err(Error::InvalidFrame("the period must be positive".into()));
        }
        frame.validate()?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                frame,
                period,
                stopped: false,
                sent: 0,
                last_error: None,
            }),
            wake: Condvar::new(),
        });
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("hilspy-periodic".into())
            .spawn(move || run(bus, worker, duration))?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Replace the frame sent from the next period on.
    pub fn modify(&self, frame: Frame) -> Result<()> {
        frame.validate()?;
        self.shared.state.lock().unwrap().frame = frame;
        Ok(())
    }

    pub fn set_period(&self, period: Duration) -> Result<()> {
        if period.is_zero() {
            return Err(Error::InvalidFrame("the period must be positive".into()));
        }
        self.shared.state.lock().unwrap().period = period;
        self.shared.wake.notify_all();
        Ok(())
    }

    pub fn period(&self) -> Duration {
        self.shared.state.lock().unwrap().period
    }

    pub fn frame(&self) -> Frame {
        self.shared.state.lock().unwrap().frame.clone()
    }

    /// Frames sent so far.
    pub fn sent_count(&self) -> u64 {
        self.shared.state.lock().unwrap().sent
    }

    /// The last send error (the task keeps running after one).
    pub fn last_error(&self) -> Option<String> {
        self.shared.state.lock().unwrap().last_error.clone()
    }

    pub fn is_running(&self) -> bool {
        !self.shared.state.lock().unwrap().stopped
            && self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Stop sending and wait for the thread to exit.
    pub fn stop(&mut self) {
        self.shared.state.lock().unwrap().stopped = true;
        self.shared.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for PeriodicTask {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(bus: Bus, shared: Arc<Shared>, duration: Option<Duration>) {
    let start = Instant::now();
    let end = duration.map(|d| start + d);
    let mut next = start;
    let mut state = shared.state.lock().unwrap();
    loop {
        if state.stopped || bus.is_closed() {
            break;
        }
        let now = Instant::now();
        if end.is_some_and(|end| now >= end) {
            break;
        }
        if now >= next {
            let frame = state.frame.clone();
            drop(state);
            let result = bus.send(&frame);
            state = shared.state.lock().unwrap();
            match result {
                Ok(()) => state.sent += 1,
                Err(e) => state.last_error = Some(e.to_string()),
            }
            next += state.period;
            let now = Instant::now();
            if next + state.period < now {
                next = now;
            }
            continue;
        }
        let wait_until = end.map_or(next, |end| end.min(next));
        state = shared
            .wake
            .wait_timeout(state, wait_until.saturating_duration_since(now))
            .unwrap()
            .0;
    }
    state.stopped = true;
}

#[cfg(test)]
mod tests {
    use super::super::{Bus, BusConfig, Frame};
    use std::time::Duration;

    #[test]
    fn sends_on_period_and_stops() {
        let tx = Bus::open(BusConfig::new("virtual", "t-periodic")).unwrap();
        let rx = Bus::open(BusConfig::new("virtual", "t-periodic")).unwrap();
        let mut task = tx
            .send_periodic(
                Frame::new(0x10, &[1], false).unwrap(),
                Duration::from_millis(10),
                None,
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(105));
        task.modify(Frame::new(0x10, &[2], false).unwrap()).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        task.stop();
        let sent = task.sent_count();
        assert!((10..=20).contains(&sent), "sent {sent}");
        let mut frames = Vec::new();
        while let Some(f) = rx.recv(Some(Duration::ZERO)).unwrap() {
            frames.push(f);
        }
        assert_eq!(frames.len() as u64, sent);
        assert_eq!(frames.first().unwrap().data, vec![1]);
        assert_eq!(frames.last().unwrap().data, vec![2]);
    }

    #[test]
    fn duration_ends_the_task() {
        let tx = Bus::open(BusConfig::new("virtual", "t-periodic-duration")).unwrap();
        let task = tx
            .send_periodic(
                Frame::new(0x10, &[], false).unwrap(),
                Duration::from_millis(5),
                Some(Duration::from_millis(30)),
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(80));
        assert!(!task.is_running());
        assert!(
            (5..=8).contains(&task.sent_count()),
            "{}",
            task.sent_count()
        );
    }
}
