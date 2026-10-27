//! A simulated loss of a channel's link, for fault injection (see
//! [`crate::Channel::disconnect`]).
//!
//! The link is cut above the transport: while it is down, sends are dropped
//! after encoding (a codec's counters keep counting, as an ECU's do when its
//! bus is cut) and received packets are dropped as the channel reads them,
//! so the transport itself (a socket, a TCP connection) stays open and the
//! peer sees the line go quiet. Once the receive side is back, packets the
//! transport stamped with a wall-clock arrival time inside the outage are
//! dropped too: they arrived while nobody was reading.
//!
//! A cut may name the messages it stops (`VehicleSpeed`, a LiDAR's `scan`):
//! then only those are lost, by name, before encoding and after decoding, and
//! the rest of the link carries on.

use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::clock::{to_ns, Clock};
use crate::error::{Error, Result};
use crate::message::now_timestamp;

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
    fn sends(self) -> bool {
        matches!(self, Self::Send | Self::Both)
    }

    fn receives(self) -> bool {
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

/// When a cut ends.
#[derive(Clone, Copy)]
enum End {
    /// At [`Outage::restore`].
    Never,
    /// On the wall clock; `unix` is the same instant as a Unix time.
    Wall { at: Instant, unix: f64 },
    /// When the virtual clock reaches this time (ns).
    Clock(i64),
}

/// The messages a cut stops: by name, or (`None`) everything.
pub(crate) type Only = Option<Arc<BTreeSet<String>>>;

#[derive(Clone)]
struct Cut {
    end: End,
    only: Only,
    /// Tells a cut from the one it replaced, for the clock task ending it.
    id: u64,
}

/// What a received packet loses to an outage.
pub(crate) enum Loss {
    Nothing,
    Everything,
    /// The messages with these names.
    Named(Arc<BTreeSet<String>>),
}

impl Loss {
    fn of(only: &Only) -> Self {
        match only {
            None => Self::Everything,
            Some(names) => Self::Named(Arc::clone(names)),
        }
    }
}

#[derive(Default)]
struct State {
    send: Option<Cut>,
    recv: Option<Cut>,
    /// Unix time the receive side has been down since.
    recv_since: f64,
    /// The last receive outage that ended, as Unix times `[from, to)`,
    /// with what it stopped.
    recv_window: Option<(f64, f64, Only)>,
    next_id: u64,
}

/// A channel's link state.
pub(crate) struct Outage {
    state: Mutex<State>,
    clock: Option<Clock>,
}

impl Outage {
    pub(crate) fn new(clock: Option<Clock>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            clock,
        })
    }

    /// Cut `direction` for `duration` (until [`Outage::restore`] without
    /// one), measured on the virtual clock if there is one, stopping the
    /// messages named in `only` (everything without). A side already down
    /// gets the new end and messages.
    pub(crate) fn cut(
        self: &Arc<Self>,
        direction: LinkDirection,
        duration: Option<Duration>,
        only: Option<BTreeSet<String>>,
    ) -> Result<()> {
        let end = match (duration, &self.clock) {
            (None, _) => End::Never,
            (Some(d), Some(clock)) => {
                End::Clock(clock.now_ns().saturating_add(to_ns(d.as_secs_f64())?))
            }
            (Some(d), None) => End::Wall {
                at: Instant::now() + d,
                unix: now_timestamp() + d.as_secs_f64(),
            },
        };
        let mut s = self.state.lock().unwrap();
        let id = s.next_id;
        s.next_id += 1;
        let cut = Some(Cut {
            end,
            only: only.map(Arc::new),
            id,
        });
        if direction.sends() {
            s.send = cut.clone();
        }
        if direction.receives() {
            if s.recv.is_none() {
                s.recv_since = now_timestamp();
            }
            s.recv = cut;
        }
        drop(s);
        // A cut on the clock ends as the clock passes its end, so the wall
        // time it ended at (for the arrivals it drops) is when that happened.
        if let (End::Clock(at), Some(clock)) = (end, &self.clock) {
            let outage = Arc::clone(self);
            clock.schedule(
                at,
                Box::new(move |_: &Clock, _| {
                    let mut s = outage.state.lock().unwrap();
                    if s.send.as_ref().is_some_and(|c| c.id == id) {
                        s.send = None;
                    }
                    if s.recv.as_ref().is_some_and(|c| c.id == id) {
                        Self::end_recv(&mut s, now_timestamp());
                    }
                }),
            );
        }
        Ok(())
    }

    /// Bring `direction` back up.
    pub(crate) fn restore(&self, direction: LinkDirection) {
        let mut s = self.state.lock().unwrap();
        if direction.sends() {
            s.send = None;
        }
        if direction.receives() && s.recv.is_some() {
            Self::end_recv(&mut s, now_timestamp());
        }
    }

    fn end_recv(s: &mut State, unix: f64) {
        if let Some(cut) = s.recv.take() {
            s.recv_window = Some((s.recv_since, unix, cut.only));
        }
    }

    /// End the cuts whose time has come.
    fn expire(&self, s: &mut State) {
        let now_ns = self.clock.as_ref().map(Clock::now_ns);
        let ended = |cut: &Option<Cut>| -> Option<f64> {
            match cut.as_ref()?.end {
                End::Never => None,
                End::Wall { at, unix } => (Instant::now() >= at).then_some(unix),
                End::Clock(at) => now_ns.is_some_and(|n| n >= at).then(now_timestamp),
            }
        };
        if ended(&s.send).is_some() {
            s.send = None;
        }
        if let Some(unix) = ended(&s.recv) {
            Self::end_recv(s, unix);
        }
    }

    /// Whether `direction` is down (either side, for [`LinkDirection::Both`]).
    pub(crate) fn is_down(&self, direction: LinkDirection) -> bool {
        let mut s = self.state.lock().unwrap();
        self.expire(&mut s);
        (direction.sends() && s.send.is_some()) || (direction.receives() && s.recv.is_some())
    }

    /// Whether a message named `name` is lost on the way out.
    pub(crate) fn drops_sent(&self, name: Option<&str>) -> bool {
        let mut s = self.state.lock().unwrap();
        self.expire(&mut s);
        match s.send.as_ref().map(|c| &c.only) {
            None => false,
            Some(None) => true,
            Some(Some(names)) => name.is_some_and(|n| names.contains(n)),
        }
    }

    /// What a received packet stamped `arrival` (a Unix time, as live
    /// transports stamp packets) loses: what the receive side being down
    /// stops, or else what the last receive outage stopped if it arrived
    /// during it.
    pub(crate) fn received(&self, arrival: Option<f64>) -> Loss {
        let mut s = self.state.lock().unwrap();
        self.expire(&mut s);
        if let Some(cut) = &s.recv {
            return Loss::of(&cut.only);
        }
        match (&s.recv_window, arrival) {
            (Some((from, to, only)), Some(t)) if t >= *from && t < *to => Loss::of(only),
            _ => Loss::Nothing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lost(o: &Outage, arrival: Option<f64>) -> bool {
        matches!(o.received(arrival), Loss::Everything)
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
        assert!(!o.is_down(LinkDirection::Recv) && !lost(&o, None));
        o.cut(LinkDirection::Recv, None, None).unwrap();
        assert!(lost(&o, None));
        o.restore(LinkDirection::Send);
        assert!(!o.is_down(LinkDirection::Send) && o.is_down(LinkDirection::Recv));
        o.restore(LinkDirection::Both);
        assert!(!o.is_down(LinkDirection::Both));
    }

    #[test]
    fn arrivals_during_the_outage_stay_lost() {
        let o = Outage::new(None);
        let before = now_timestamp();
        std::thread::sleep(Duration::from_millis(2));
        o.cut(LinkDirection::Recv, None, None).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let during = now_timestamp();
        std::thread::sleep(Duration::from_millis(2));
        o.restore(LinkDirection::Recv);
        // Down, everything goes; back up, only what arrived meanwhile.
        assert!(!lost(&o, Some(before)));
        assert!(lost(&o, Some(during)));
        assert!(!lost(&o, Some(now_timestamp())));
        assert!(!lost(&o, None));
    }

    #[test]
    fn wall_cuts_end_by_themselves() {
        let o = Outage::new(None);
        o.cut(LinkDirection::Both, Some(Duration::from_millis(30)), None)
            .unwrap();
        assert!(o.is_down(LinkDirection::Both));
        std::thread::sleep(Duration::from_millis(40));
        assert!(!o.is_down(LinkDirection::Both));
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
        assert_eq!(clock.pending(), 0);
    }

    #[test]
    fn a_new_cut_replaces_the_old_end() {
        let clock = Clock::new(0.0).unwrap();
        let o = Outage::new(Some(clock.clone()));
        o.cut(LinkDirection::Send, Some(Duration::from_millis(100)), None)
            .unwrap();
        o.cut(LinkDirection::Send, None, None).unwrap();
        // The first cut's end does not end the second.
        clock.advance(1.0).unwrap();
        assert!(o.is_down(LinkDirection::Send));
        o.restore(LinkDirection::Send);
        o.cut(LinkDirection::Send, Some(Duration::from_millis(100)), None)
            .unwrap();
        o.restore(LinkDirection::Send);
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
        o.cut(LinkDirection::Both, None, names(&["scan", "VehicleSpeed"]))
            .unwrap();
        assert!(o.is_down(LinkDirection::Both));
        assert!(o.drops_sent(Some("scan")));
        assert!(!o.drops_sent(Some("SteeringCmd")) && !o.drops_sent(None));
        let Loss::Named(stopped) = o.received(None) else {
            panic!("a named cut stops named messages");
        };
        assert!(stopped.contains("VehicleSpeed") && stopped.len() == 2);
        // A new cut replaces the names; restoring ends it.
        o.cut(LinkDirection::Send, None, None).unwrap();
        assert!(o.drops_sent(Some("SteeringCmd")) && o.drops_sent(None));
        o.restore(LinkDirection::Both);
        assert!(!o.drops_sent(Some("scan")));
        assert!(matches!(o.received(None), Loss::Nothing));
    }
}
