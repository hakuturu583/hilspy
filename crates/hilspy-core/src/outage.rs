//! A simulated loss of a channel's link, for fault injection (see
//! [`crate::Channel::disconnect`]).
//!
//! The link is cut above the transport: while it is down, sends are dropped
//! after encoding (a codec's counters keep counting, as an ECU's do when its
//! bus is cut) and received packets are dropped as the channel reads them,
//! so the transport itself (a socket, a TCP connection) stays open and the
//! peer sees the line go quiet. Once the receive side is back, the next
//! receive first drains what the transport buffered meanwhile: it arrived
//! while the line was down, whenever the transport stamped it.
//!
//! A cut may name the messages it stops (`VehicleSpeed`, a LiDAR's `scan`):
//! then only those are lost -- after encoding on the way out, after decoding
//! on the way in -- and the rest of the link carries on.

use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::clock::Clock;
use crate::error::{Error, Result};

/// Which way of a channel's link an outage cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkDirection {
    /// What the channel sends.
    Send,
    /// What the channel receives.
    Recv,
    Both,
}

impl LinkDirection {
    pub(crate) fn sends(self) -> bool {
        matches!(self, Self::Send | Self::Both)
    }

    pub(crate) fn receives(self) -> bool {
        matches!(self, Self::Recv | Self::Both)
    }
}

impl FromStr for LinkDirection {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "send" => Ok(Self::Send),
            "recv" => Ok(Self::Recv),
            "both" => Ok(Self::Both),
            _ => Err(Error::Config(format!(
                "unknown direction {s:?}; expected send, recv or both"
            ))),
        }
    }
}

/// What an outage stops.
#[derive(Clone, Debug)]
pub(crate) enum Loss {
    Nothing,
    Everything,
    /// The messages with these names.
    Named(Arc<BTreeSet<String>>),
}

impl Loss {
    /// Whether a message named `name` is lost.
    pub(crate) fn drops(&self, name: Option<&str>) -> bool {
        match self {
            Self::Nothing => false,
            Self::Everything => true,
            Self::Named(names) => name.is_some_and(|n| names.contains(n)),
        }
    }

    /// What `self` and `other` stop together.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Nothing, l) | (l, Self::Nothing) => l,
            (Self::Named(a), Self::Named(b)) => {
                Self::Named(Arc::new(a.union(&b).cloned().collect()))
            }
            _ => Self::Everything,
        }
    }
}

/// When a cut ends.
#[derive(Clone, Copy)]
enum End {
    /// At [`Outage::restore`].
    Never,
    Wall(Instant),
    /// When the virtual clock reaches this time (ns).
    Clock(i64),
}

#[derive(Clone)]
struct Cut {
    /// Never [`Loss::Nothing`].
    loss: Loss,
    end: End,
}

struct State {
    send: Option<Cut>,
    recv: Option<Cut>,
    /// What the receive side stopped before it came back: the next receive
    /// drops it from what the transport buffered.
    drain: Loss,
}

/// A channel's link state.
pub(crate) struct Outage {
    state: Mutex<State>,
    /// Whether anything is cut or to drain: every packet sent and received
    /// checks it without locking.
    active: AtomicBool,
    clock: Option<Clock>,
}

impl Outage {
    pub(crate) fn new(clock: Option<Clock>) -> Self {
        Self {
            state: Mutex::new(State {
                send: None,
                recv: None,
                drain: Loss::Nothing,
            }),
            active: AtomicBool::new(false),
            clock,
        }
    }

    /// Cut `direction` for `duration` (until [`Outage::restore`] without
    /// one), measured on the virtual clock if there is one, stopping the
    /// messages named in `only` (everything without). A side already down
    /// gets the new end and messages. Returns what the receive side stops.
    pub(crate) fn cut(
        &self,
        direction: LinkDirection,
        duration: Option<Duration>,
        only: Option<BTreeSet<String>>,
    ) -> Result<Loss> {
        let loss = match only {
            None => Loss::Everything,
            Some(names) if names.is_empty() => {
                return Err(Error::Config(
                    "a cut must name at least one message (or none, for everything)".into(),
                ))
            }
            Some(names) => Loss::Named(Arc::new(names)),
        };
        let end = match (duration, &self.clock) {
            (None, _) => End::Never,
            (Some(d), Some(clock)) => End::Clock(clock.deadline_after(d)),
            // Too far to represent: until reconnected.
            (Some(d), None) => Instant::now().checked_add(d).map_or(End::Never, End::Wall),
        };
        let cut = Some(Cut { loss, end });
        let mut s = self.state.lock().unwrap();
        if direction.sends() {
            s.send = cut.clone();
        }
        if direction.receives() {
            s.recv = cut;
        }
        self.active.store(true, Ordering::Release);
        Ok(s.recv.as_ref().map_or(Loss::Nothing, |c| c.loss.clone()))
    }

    /// Bring `direction` back up.
    pub(crate) fn restore(&self, direction: LinkDirection) {
        let mut s = self.state.lock().unwrap();
        if direction.sends() {
            s.send = None;
        }
        if direction.receives() {
            Self::end_recv(&mut s);
        }
        self.settle(&s);
    }

    fn end_recv(s: &mut State) {
        if let Some(cut) = s.recv.take() {
            s.drain = std::mem::replace(&mut s.drain, Loss::Nothing).and(cut.loss);
        }
    }

    /// Clear the fast-path flag once nothing is left.
    fn settle(&self, s: &State) {
        if s.send.is_none() && s.recv.is_none() && matches!(s.drain, Loss::Nothing) {
            self.active.store(false, Ordering::Release);
        }
    }

    /// The state with the cuts whose time has come ended, or `None` when
    /// nothing is cut (found without locking).
    fn current(&self) -> Option<MutexGuard<'_, State>> {
        if !self.active.load(Ordering::Acquire) {
            return None;
        }
        let mut s = self.state.lock().unwrap();
        let ended = |cut: &Option<Cut>| match cut.as_ref().map(|c| c.end) {
            Some(End::Wall(at)) => Instant::now() >= at,
            Some(End::Clock(at)) => self.clock.as_ref().is_some_and(|c| c.now_ns() >= at),
            Some(End::Never) | None => false,
        };
        if ended(&s.send) {
            s.send = None;
        }
        if ended(&s.recv) {
            Self::end_recv(&mut s);
        }
        self.settle(&s);
        Some(s)
    }

    /// Whether `direction` is down (either side, for [`LinkDirection::Both`]).
    pub(crate) fn is_down(&self, direction: LinkDirection) -> bool {
        self.current().is_some_and(|s| {
            (direction.sends() && s.send.is_some()) || (direction.receives() && s.recv.is_some())
        })
    }

    /// Whether a packet of a message named `name` is lost on the way out.
    pub(crate) fn drops_sent(&self, name: Option<&str>) -> bool {
        self.current()
            .and_then(|s| s.send.as_ref().map(|c| c.loss.drops(name)))
            .unwrap_or(false)
    }

    /// What a received packet loses.
    pub(crate) fn received(&self) -> Loss {
        self.current()
            .and_then(|s| s.recv.as_ref().map(|c| c.loss.clone()))
            .unwrap_or(Loss::Nothing)
    }

    /// What to drop of what the transport buffered while the receive side
    /// was down, once; `None` when there is nothing to drain.
    pub(crate) fn take_drain(&self) -> Option<Loss> {
        let mut s = self.current()?;
        let drain = std::mem::replace(&mut s.drain, Loss::Nothing);
        self.settle(&s);
        (!matches!(drain, Loss::Nothing)).then_some(drain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lost(o: &Outage) -> bool {
        matches!(o.received(), Loss::Everything)
    }

    fn names(list: &[&str]) -> Option<BTreeSet<String>> {
        Some(list.iter().map(|n| (*n).to_owned()).collect())
    }

    #[test]
    fn directions_parse() {
        assert_eq!(
            "send".parse::<LinkDirection>().unwrap(),
            LinkDirection::Send
        );
        assert_eq!(
            "recv".parse::<LinkDirection>().unwrap(),
            LinkDirection::Recv
        );
        assert_eq!(
            "both".parse::<LinkDirection>().unwrap(),
            LinkDirection::Both
        );
        let e = "tx".parse::<LinkDirection>().unwrap_err();
        assert!(e.to_string().contains("send, recv or both"), "{e}");
    }

    #[test]
    fn cuts_each_side_until_restored() {
        let o = Outage::new(None);
        assert!(!o.is_down(LinkDirection::Both));
        o.cut(LinkDirection::Send, None, None).unwrap();
        assert!(o.is_down(LinkDirection::Send) && o.is_down(LinkDirection::Both));
        assert!(!o.is_down(LinkDirection::Recv) && !lost(&o));
        o.cut(LinkDirection::Recv, None, None).unwrap();
        assert!(lost(&o));
        o.restore(LinkDirection::Send);
        assert!(!o.is_down(LinkDirection::Send) && o.is_down(LinkDirection::Recv));
        o.restore(LinkDirection::Both);
        assert!(!o.is_down(LinkDirection::Both));
    }

    #[test]
    fn what_was_buffered_meanwhile_is_drained_once() {
        let o = Outage::new(None);
        assert!(o.take_drain().is_none());
        o.cut(LinkDirection::Recv, None, names(&["a"])).unwrap();
        o.restore(LinkDirection::Recv);
        o.cut(LinkDirection::Recv, None, names(&["b"])).unwrap();
        o.restore(LinkDirection::Recv);
        // Two outages before anyone read: what either stopped goes.
        let Some(Loss::Named(drain)) = o.take_drain() else {
            panic!("named cuts drain named messages");
        };
        assert!(drain.contains("a") && drain.contains("b"));
        assert!(o.take_drain().is_none());
        o.cut(LinkDirection::Recv, None, None).unwrap();
        o.restore(LinkDirection::Both);
        assert!(matches!(o.take_drain(), Some(Loss::Everything)));
        assert!(!o.active.load(Ordering::Acquire));
    }

    #[test]
    fn wall_cuts_end_by_themselves() {
        let o = Outage::new(None);
        o.cut(LinkDirection::Both, Some(Duration::from_millis(30)), None)
            .unwrap();
        assert!(o.is_down(LinkDirection::Both));
        std::thread::sleep(Duration::from_millis(40));
        assert!(!o.is_down(LinkDirection::Both));
        assert!(matches!(o.take_drain(), Some(Loss::Everything)));
    }

    #[test]
    fn clock_cuts_end_as_the_clock_passes() {
        let clock = Clock::new(0.0).unwrap();
        let o = Outage::new(Some(clock.clone()));
        o.cut(LinkDirection::Both, Some(Duration::from_millis(500)), None)
            .unwrap();
        // Real time does not end it.
        std::thread::sleep(Duration::from_millis(20));
        assert!(o.is_down(LinkDirection::Both));
        clock.advance(0.4).unwrap();
        assert!(o.is_down(LinkDirection::Both));
        clock.advance(0.1).unwrap();
        assert!(!o.is_down(LinkDirection::Both));
        // Nothing is left on the clock.
        assert_eq!(clock.pending(), 0);
    }

    #[test]
    fn a_new_cut_replaces_the_old_end() {
        let clock = Clock::new(0.0).unwrap();
        let o = Outage::new(Some(clock.clone()));
        o.cut(LinkDirection::Send, Some(Duration::from_millis(100)), None)
            .unwrap();
        o.cut(LinkDirection::Send, None, None).unwrap();
        clock.advance(1.0).unwrap();
        assert!(o.is_down(LinkDirection::Send));
        o.cut(LinkDirection::Send, Some(Duration::from_millis(500)), None)
            .unwrap();
        clock.advance(0.2).unwrap();
        assert!(o.is_down(LinkDirection::Send));
        clock.advance(0.3).unwrap();
        assert!(!o.is_down(LinkDirection::Send));
    }

    #[test]
    fn a_cut_can_stop_only_named_messages() {
        let o = Outage::new(None);
        let loss = o
            .cut(LinkDirection::Both, None, names(&["scan", "VehicleSpeed"]))
            .unwrap();
        assert!(loss.drops(Some("VehicleSpeed")) && !loss.drops(None));
        assert!(o.is_down(LinkDirection::Both));
        assert!(o.drops_sent(Some("scan")));
        assert!(!o.drops_sent(Some("SteeringCmd")) && !o.drops_sent(None));
        // A new cut replaces the names; restoring ends it.
        o.cut(LinkDirection::Send, None, None).unwrap();
        assert!(o.drops_sent(Some("SteeringCmd")) && o.drops_sent(None));
        o.restore(LinkDirection::Both);
        assert!(!o.drops_sent(Some("scan")));
        assert!(matches!(o.received(), Loss::Nothing));
    }

    #[test]
    fn empty_names_and_huge_durations() {
        let o = Outage::new(None);
        assert!(o.cut(LinkDirection::Both, None, names(&[])).is_err());
        o.cut(LinkDirection::Both, Some(Duration::MAX), None)
            .unwrap();
        assert!(o.is_down(LinkDirection::Both));
    }
}
