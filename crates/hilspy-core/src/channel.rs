//! A channel: a transport and a codec composed, sending and receiving named
//! messages.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::clock::{to_ns, Clock};
use crate::codec::{self, Codec};
use crate::error::{Error, Result};
use crate::message::{Message, Packet};
use crate::transport::{self, Transport};
use crate::value::Options;

struct Inner {
    transport: Arc<dyn Transport>,
    codec: Mutex<Box<dyn Codec>>,
    /// Held for a whole send (encode and every packet), so the packets of
    /// concurrent sends do not interleave; decoding does not take it.
    sending: Mutex<()>,
    pending: Mutex<VecDeque<Message>>,
    closed: AtomicBool,
    /// The error of a packet sent on the clock after its message's send
    /// returned, reported by the next send.
    deferred_error: Mutex<Option<Error>>,
    /// Virtual time the channel runs on, if any.
    clock: Option<Clock>,
}

impl Inner {
    fn send(self: &Arc<Self>, message: &Message) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let _sending = self.sending.lock().unwrap();
        // On a clock, a message to encode is stamped with its time, which
        // codecs use for the times they write (SBF TOW, a LiDAR scan).
        let stamped;
        let message = match (&self.clock, &message.name) {
            (Some(clock), Some(_)) => {
                let mut m = message.clone();
                m.packet.get_or_insert_with(Packet::default).timestamp = Some(clock.now());
                stamped = m;
                &stamped
            }
            _ => message,
        };
        let (packets, paced) = match (&message.name, &message.packet) {
            (None, Some(p)) => (vec![p.clone()], false),
            (None, None) => {
                return Err(Error::encode("a message needs a name or data"));
            }
            (Some(_), _) => {
                let mut codec = self.codec.lock().unwrap();
                (codec.encode(message)?, codec.paced())
            }
        };
        let first = packets.first().and_then(|p| p.timestamp);
        if let (Some(clock), true) = (&self.clock, paced) {
            // Later packets are sent as the clock passes their times.
            let now = clock.now_ns();
            for p in packets {
                let delay = match (first, p.timestamp) {
                    // Whole microseconds: the difference of two absolute
                    // Unix times in f64 is only good to a fraction of one.
                    (Some(t0), Some(t)) if (t - t0).is_finite() => {
                        to_ns(((t - t0).clamp(0.0, 3600.0) * 1e6).round() * 1e-6)?
                    }
                    _ => 0,
                };
                if delay == 0 {
                    self.transport.send(&p)?;
                } else {
                    let inner = Arc::clone(self);
                    clock.schedule(
                        now + delay,
                        Box::new(move |_: &Clock, _| {
                            if !inner.closed.load(Ordering::Acquire) {
                                // A failure has no caller to report to;
                                // it is recorded for the next send.
                                if let Err(e) = inner.transport.send(&p) {
                                    *inner.deferred_error.lock().unwrap() = Some(e);
                                }
                            }
                        }),
                    );
                }
            }
            return match self.deferred_error.lock().unwrap().take() {
                Some(e) => Err(e),
                None => Ok(()),
            };
        }
        let start = Instant::now();
        for p in &packets {
            if let (true, Some(t0), Some(t)) = (paced, first, p.timestamp) {
                let delay = t - t0;
                if delay.is_finite() {
                    self.wait_until(start + Duration::from_secs_f64(delay.clamp(0.0, 3600.0)))?;
                }
            }
            self.transport.send(p)?;
        }
        Ok(())
    }

    /// Sleep until `deadline`, returning early with `Closed` on close.
    fn wait_until(&self, deadline: Instant) -> Result<()> {
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(());
            }
            std::thread::sleep(left.min(PACE_POLL));
        }
    }
}

/// How often a paced send checks for close.
const PACE_POLL: Duration = Duration::from_millis(20);

/// A transport and a codec composed. Cloning shares the channel.
#[derive(Clone)]
pub struct Channel {
    inner: Arc<Inner>,
}

impl Channel {
    /// Open the transport `url` with the codec `codec` (`raw`, `sbf`,
    /// `dbc:FILE`, ...) configured by `options`.
    pub fn open(url: &str, codec: &str, options: Options) -> Result<Self> {
        Self::open_on(url, codec, options, None)
    }

    /// [`Channel::open`] on a virtual clock (see [`crate::clock`]).
    pub fn open_on(url: &str, codec: &str, options: Options, clock: Option<Clock>) -> Result<Self> {
        let codec = codec::open(codec, options)?;
        let transport = transport::open(url)?;
        Ok(Self::with_clock(transport, codec, clock))
    }

    pub fn new(transport: Arc<dyn Transport>, codec: Box<dyn Codec>) -> Self {
        Self::with_clock(transport, codec, None)
    }

    pub fn with_clock(
        transport: Arc<dyn Transport>,
        codec: Box<dyn Codec>,
        clock: Option<Clock>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                transport,
                codec: Mutex::new(codec),
                sending: Mutex::new(()),
                pending: Mutex::new(VecDeque::new()),
                closed: AtomicBool::new(false),
                deferred_error: Mutex::new(None),
                clock,
            }),
        }
    }

    /// The virtual clock the channel runs on.
    pub fn clock(&self) -> Option<&Clock> {
        self.inner.clock.as_ref()
    }

    /// Encode and send a message (a nameless one sends its packet as is).
    pub fn send(&self, message: &Message) -> Result<()> {
        self.inner.send(message)
    }

    /// The next decoded message, or `None` once `timeout` passes.
    pub fn recv(&self, timeout: Option<Duration>) -> Result<Option<Message>> {
        let inner = &self.inner;
        if inner.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        if let Some(m) = inner.pending.lock().unwrap().pop_front() {
            return Ok(Some(m));
        }
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if inner.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            let Some(mut packet) = inner.transport.recv(remaining)? else {
                return Ok(None);
            };
            if let Some(clock) = &inner.clock {
                packet.timestamp = Some(clock.now());
            }
            let mut messages: VecDeque<Message> =
                inner.codec.lock().unwrap().decode(packet)?.into();
            if let Some(first) = messages.pop_front() {
                inner.pending.lock().unwrap().extend(messages);
                return Ok(Some(first));
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
        }
    }

    /// Send `message` every `period` from a background thread (for
    /// `duration`, or until stopped). It is encoded on every send, so
    /// time-stamping codecs stamp each one. On a virtual clock the sends
    /// happen as the clock advances past them instead (the first at once).
    pub fn every(
        &self,
        period: Duration,
        message: Message,
        duration: Option<Duration>,
    ) -> Result<Periodic> {
        Periodic::start(Arc::clone(&self.inner), period, message, duration)
    }

    /// Receive on a background thread, handing every message (or receive
    /// error) to `deliver` as it arrives, until `deliver` returns `false`,
    /// the subscription is stopped or the channel closes (then `deliver` gets
    /// `Err(Error::Closed)` once). One thread per subscribed channel lets a
    /// program serve many buses at once without polling them in turn.
    ///
    /// The subscription is the channel's only consumer meanwhile: `recv`
    /// from elsewhere would take messages from it.
    pub fn subscribe<F>(&self, mut deliver: F) -> Result<Subscription>
    where
        F: FnMut(Result<Message>) -> bool + Send + 'static,
    {
        if self.is_closed() {
            return Err(Error::Closed);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let channel = self.clone();
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("hilspy-subscription".into())
            .spawn(move || {
                let mut backoff = Duration::ZERO;
                while !flag.load(Ordering::Acquire) {
                    match channel.recv(Some(SUBSCRIPTION_POLL)) {
                        Ok(Some(m)) => {
                            backoff = Duration::ZERO;
                            if !deliver(Ok(m)) {
                                break;
                            }
                        }
                        Ok(None) => backoff = Duration::ZERO,
                        Err(Error::Closed) => {
                            deliver(Err(Error::Closed));
                            break;
                        }
                        // A packet that does not decode does not end the
                        // stream; an error that repeats at once (a failing
                        // transport) is retried ever more slowly, so it
                        // neither spins nor floods the consumer.
                        Err(e) => {
                            if !deliver(Err(e)) {
                                break;
                            }
                            sleep_unless_stopped(&flag, backoff);
                            backoff = (backoff * 2).clamp(ERROR_BACKOFF_MIN, ERROR_BACKOFF_MAX);
                        }
                    }
                }
            })?;
        Ok(Subscription {
            stop,
            thread: Mutex::new(Some(thread)),
        })
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    /// Close the channel: blocked receivers return `Closed`, periodic sends
    /// stop.
    pub fn close(&self) {
        if !self.inner.closed.swap(true, Ordering::AcqRel) {
            self.inner.transport.close();
        }
    }
}

/// How often a subscription thread checks whether it was stopped.
const SUBSCRIPTION_POLL: Duration = Duration::from_millis(50);
/// Delay before retrying after consecutive receive errors (doubling from the
/// first to the second value; the first error is retried at once).
const ERROR_BACKOFF_MIN: Duration = Duration::from_millis(10);
const ERROR_BACKOFF_MAX: Duration = Duration::from_secs(1);

/// Sleep `total`, waking early once `stop` is set.
fn sleep_unless_stopped(stop: &AtomicBool, total: Duration) {
    let end = Instant::now() + total;
    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        if now >= end {
            break;
        }
        std::thread::sleep((end - now).min(SUBSCRIPTION_POLL));
    }
}

/// A running [`Channel::subscribe`]; dropping it stops it.
pub struct Subscription {
    stop: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Subscription {
    /// Stop delivering and wait for the receive thread (idempotent; at most
    /// one poll interval, plus a delivery in progress).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        let thread = self.thread.lock().unwrap().take();
        if let Some(t) = thread {
            if t.thread().id() != std::thread::current().id() {
                let _ = t.join();
            }
        }
    }

    pub fn is_running(&self) -> bool {
        self.thread
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|t| !t.is_finished())
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.stop();
    }
}

struct State {
    message: Message,
    period: Duration,
    stopped: bool,
    sent: u64,
    last_error: Option<String>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

/// A message sent on a fixed period from a background thread (the cyclic
/// messages an ECU emits). Deadlines are absolute so the period does not
/// drift; a deadline missed by more than a period is skipped, not burst.
/// Dropping it stops it.
pub struct Periodic {
    shared: Arc<Shared>,
    /// Driven by a virtual clock rather than a thread.
    on_clock: bool,
    /// Taken by the first `stop`, which joins it outside the lock.
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Periodic {
    fn start(
        inner: Arc<Inner>,
        period: Duration,
        message: Message,
        duration: Option<Duration>,
    ) -> Result<Self> {
        if period.is_zero() {
            return Err(Error::Config("the period must be positive".into()));
        }
        // The first send happens here, so a message that cannot be sent
        // fails the call (rather than a dry-run encode, which would cost a
        // stateful codec a value: an E2E counter, a sequence number).
        let start = Instant::now();
        let now = inner.clock.as_ref().map(Clock::now_ns);
        inner.send(&message)?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                message,
                period,
                stopped: false,
                sent: 1,
                last_error: None,
            }),
            wake: Condvar::new(),
        });
        if let (Some(clock), Some(now)) = (inner.clock.clone(), now) {
            let end = duration
                .map(|d| now.saturating_add(i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)));
            schedule_next(&clock, inner, Arc::clone(&shared), end, now);
            return Ok(Self {
                shared,
                on_clock: true,
                thread: Mutex::new(None),
            });
        }
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("hilspy-periodic".into())
            .spawn(move || run(inner, worker, start, duration))?;
        Ok(Self {
            shared,
            on_clock: false,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Send `message` instead from the next period on.
    pub fn update(&self, message: Message) {
        self.shared.state.lock().unwrap().message = message;
    }

    pub fn set_period(&self, period: Duration) -> Result<()> {
        if period.is_zero() {
            return Err(Error::Config("the period must be positive".into()));
        }
        self.shared.state.lock().unwrap().period = period;
        self.shared.wake.notify_all();
        Ok(())
    }

    pub fn period(&self) -> Duration {
        self.shared.state.lock().unwrap().period
    }

    pub fn message(&self) -> Message {
        self.shared.state.lock().unwrap().message.clone()
    }

    pub fn sent_count(&self) -> u64 {
        self.shared.state.lock().unwrap().sent
    }

    /// The last send error (sending continues after one).
    pub fn last_error(&self) -> Option<String> {
        self.shared.state.lock().unwrap().last_error.clone()
    }

    pub fn is_running(&self) -> bool {
        !self.shared.state.lock().unwrap().stopped
            && (self.on_clock
                || self
                    .thread
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|t| !t.is_finished()))
    }

    /// Stop and wait for the sender thread (idempotent). No lock is held
    /// while joining, so other threads can query the task meanwhile.
    pub fn stop(&self) {
        self.shared.state.lock().unwrap().stopped = true;
        self.shared.wake.notify_all();
        let thread = self.thread.lock().unwrap().take();
        if let Some(t) = thread {
            let _ = t.join();
        }
    }
}

impl Drop for Periodic {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One send of a periodic message on a virtual clock, due at `at` (ns); it
/// schedules the next.
fn tick(clock: &Clock, inner: Arc<Inner>, shared: Arc<Shared>, end: Option<i64>, at: i64) {
    let mut state = shared.state.lock().unwrap();
    if state.stopped || inner.closed.load(Ordering::Acquire) || end.is_some_and(|e| at >= e) {
        state.stopped = true;
        return;
    }
    let message = state.message.clone();
    drop(state);
    let result = inner.send(&message);
    let mut state = shared.state.lock().unwrap();
    match result {
        Ok(()) => state.sent += 1,
        Err(e) => state.last_error = Some(e.to_string()),
    }
    drop(state);
    schedule_next(clock, inner, shared, end, at);
}

/// Schedule the tick after the one due at `at`, unless it falls at or after
/// the end: then stop now, not when the clock reaches it.
fn schedule_next(clock: &Clock, inner: Arc<Inner>, shared: Arc<Shared>, end: Option<i64>, at: i64) {
    let mut state = shared.state.lock().unwrap();
    let next = at.saturating_add(i64::try_from(state.period.as_nanos()).unwrap_or(i64::MAX));
    if end.is_some_and(|e| next >= e) {
        state.stopped = true;
        return;
    }
    drop(state);
    clock.schedule(
        next,
        Box::new(move |c: &Clock, due| tick(c, inner, shared, end, due)),
    );
}

/// The sender thread; the first send (at `start`) is already made.
fn run(inner: Arc<Inner>, shared: Arc<Shared>, start: Instant, duration: Option<Duration>) {
    let end = duration.map(|d| start + d);
    let mut state = shared.state.lock().unwrap();
    let mut next = start + state.period;
    loop {
        if state.stopped || inner.closed.load(Ordering::Acquire) {
            break;
        }
        let now = Instant::now();
        if end.is_some_and(|end| now >= end) {
            break;
        }
        if now >= next {
            let message = state.message.clone();
            drop(state);
            let result = inner.send(&message);
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
        let until = end.map_or(next, |end| end.min(next));
        state = shared
            .wake
            .wait_timeout(state, until.saturating_duration_since(now))
            .unwrap()
            .0;
    }
    state.stopped = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Packet;
    use crate::value::Value;

    fn dbc() -> String {
        format!(
            "dbc:{}/../../tests/data/vehicle.dbc",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn scan() -> Message {
        use crate::value::Array;
        Message::new(
            "scan",
            ["x", "y", "z"]
                .iter()
                .map(|k| ((*k).to_owned(), Value::Array(Array::F64(vec![10.0]))))
                .collect(),
        )
    }

    fn drain(rx: &Channel) -> Vec<Message> {
        let mut out = Vec::new();
        while let Some(m) = rx.recv(Some(Duration::from_millis(20))).unwrap() {
            out.push(m);
        }
        out
    }

    #[test]
    fn periodic_sends_follow_a_virtual_clock() {
        let clock = Clock::new(100.0).unwrap();
        let tx = Channel::open_on(
            "mem://t-clock-every",
            "raw",
            Options::new(),
            Some(clock.clone()),
        )
        .unwrap();
        let rx = Channel::open_on(
            "mem://t-clock-every",
            "raw",
            Options::new(),
            Some(clock.clone()),
        )
        .unwrap();
        let p = tx
            .every(
                Duration::from_millis(100),
                Message::raw(Packet::bytes(vec![1])),
                Some(Duration::from_millis(500)),
            )
            .unwrap();
        // The first send is immediate; nothing else happens until a step.
        assert_eq!(p.sent_count(), 1);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(p.sent_count(), 1);
        clock.advance(0.35).unwrap();
        assert_eq!(p.sent_count(), 4);
        // Received messages carry the virtual time they arrived at.
        let got = drain(&rx);
        assert_eq!(got.len(), 4);
        assert!(got
            .iter()
            .all(|m| (m.timestamp().unwrap() - 100.35).abs() < 1e-9));
        // The duration ends it: sends at 100.0 .. 100.4, and it stops with
        // the last one.
        clock.advance_to(100.4).unwrap();
        assert_eq!(p.sent_count(), 5);
        assert!(!p.is_running());
        assert_eq!(clock.pending(), 0);
        assert_eq!(clock.next_due(), None);

        // A duration that is not a whole number of periods.
        let p = tx
            .every(
                Duration::from_millis(100),
                Message::raw(Packet::bytes(vec![2])),
                Some(Duration::from_millis(250)),
            )
            .unwrap();
        clock.advance(0.2).unwrap();
        assert_eq!(p.sent_count(), 3);
        assert!(!p.is_running());
        assert_eq!(clock.next_due(), None);
    }

    #[test]
    fn paced_packets_follow_a_virtual_clock() {
        let clock = Clock::new(0.0).unwrap();
        let tx = Channel::open_on(
            "mem://t-clock-pace",
            "lidar:vlp16",
            Options::new(),
            Some(clock.clone()),
        )
        .unwrap();
        let rx = Channel::open("mem://t-clock-pace", "raw", Options::new()).unwrap();
        let start = Instant::now();
        tx.send(&scan()).unwrap();
        assert!(
            start.elapsed() < Duration::from_millis(50),
            "does not block"
        );
        // 75 packets 1.327 ms apart: the first goes at once, the rest as
        // the clock passes them.
        assert_eq!(drain(&rx).len(), 1);
        clock.advance(0.05).unwrap();
        assert_eq!(drain(&rx).len(), 37);
        clock.advance(0.05).unwrap();
        assert_eq!(drain(&rx).len(), 37);
        assert_eq!(clock.pending(), 0);
    }

    #[test]
    fn codecs_stamp_with_the_clock() {
        // 2024-01-01T00:00:00Z: GPS week 2295, TOW Monday 00:00:18.
        let clock = Clock::new(1_704_067_200.0).unwrap();
        let tx = Channel::open_on(
            "mem://t-clock-sbf",
            "sbf",
            Options::new(),
            Some(clock.clone()),
        )
        .unwrap();
        let rx = Channel::open("mem://t-clock-sbf", "sbf", Options::new()).unwrap();
        tx.send(&Message::new("AttEuler", vec![])).unwrap();
        clock.advance(0.5).unwrap();
        tx.send(&Message::new("AttEuler", vec![])).unwrap();
        let got = drain(&rx);
        assert_eq!(got[0].field("wnc"), Some(&Value::Int(2295)));
        assert_eq!(
            got[0].field("tow"),
            Some(&Value::Int((24 * 3600 + 18) * 1000))
        );
        assert_eq!(
            got[1].field("tow"),
            Some(&Value::Int((24 * 3600 + 18) * 1000 + 500))
        );
    }

    #[test]
    fn paced_codecs_spread_a_message_over_its_packet_times() {
        use crate::value::{Array, Value};
        let scan = Message::new(
            "scan",
            ["x", "y", "z"]
                .iter()
                .map(|k| ((*k).to_owned(), Value::Array(Array::F64(vec![10.0]))))
                .collect(),
        );
        let paced = Channel::open("mem://t-ch-pace", "lidar:vlp16", Options::new()).unwrap();
        let rx = Channel::open("mem://t-ch-pace", "raw", Options::new()).unwrap();
        let start = Instant::now();
        paced.send(&scan).unwrap();
        // 75 packets of 24 firings, 55.296 us apart: the last leaves 74 x
        // 1.327 ms after the first.
        assert!(
            start.elapsed() >= Duration::from_millis(95),
            "{:?}",
            start.elapsed()
        );
        let mut fast = Options::new();
        fast.insert("pace", Value::Bool(false));
        let burst = Channel::open("mem://t-ch-pace", "lidar:vlp16", fast).unwrap();
        let start = Instant::now();
        burst.send(&scan).unwrap();
        assert!(start.elapsed() < Duration::from_millis(80));
        let mut n = 0;
        while rx.recv(Some(Duration::from_millis(50))).unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 150);
    }

    #[test]
    fn concurrent_paced_sends_do_not_interleave() {
        use crate::value::{Array, Value};
        let scan = Message::new(
            "scan",
            ["x", "y", "z"]
                .iter()
                .map(|k| ((*k).to_owned(), Value::Array(Array::F64(vec![10.0]))))
                .collect(),
        );
        let tx = Channel::open("mem://t-ch-serial", "lidar:pandar_xt32", Options::new()).unwrap();
        let rx = Channel::open("mem://t-ch-serial", "raw", Options::new()).unwrap();
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let (tx, scan) = (tx.clone(), scan.clone());
                std::thread::spawn(move || tx.send(&scan).unwrap())
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // The UDP sequence number (u32 at 1076) counts packets in send order.
        let mut sequence = Vec::new();
        while let Some(m) = rx.recv(Some(Duration::from_millis(50))).unwrap() {
            let d = &m.packet.unwrap().data;
            sequence.push(u32::from_le_bytes([d[1076], d[1077], d[1078], d[1079]]));
        }
        assert_eq!(sequence, (0..500).collect::<Vec<_>>());
    }

    #[test]
    fn dbc_over_mem() {
        let tx = Channel::open("mem://t-ch-dbc", &dbc(), Options::new()).unwrap();
        let rx = Channel::open("mem://t-ch-dbc", &dbc(), Options::new()).unwrap();
        let raw = Channel::open("mem://t-ch-dbc", "raw", Options::new()).unwrap();
        let fields = vec![
            ("Speed".to_string(), Value::Float(123.45)),
            ("Accel".to_string(), Value::Float(-1.5)),
            ("Gear".to_string(), Value::Str("D".into())),
            ("Counter".to_string(), Value::Int(9)),
        ];
        tx.send(&Message::new("VehicleSpeed", fields)).unwrap();
        let got = rx.recv(Some(Duration::from_secs(1))).unwrap().unwrap();
        assert_eq!(got.name.as_deref(), Some("VehicleSpeed"));
        assert_eq!(got.field("Gear"), Some(&Value::Str("D".into())));
        assert_eq!(got.packet.as_ref().unwrap().can.unwrap().id, 0x100);
        let r = raw.recv(Some(Duration::from_secs(1))).unwrap().unwrap();
        assert_eq!(r.name, None);
        assert_eq!(
            r.packet.unwrap().data,
            vec![0x39, 0x30, 0xF4, 0x01, 0x03, 0x00, 0x00, 0x90]
        );
        // A nameless message is sent as is and decodes by its identifier.
        raw.send(&Message::raw(Packet::can(
            0x100,
            vec![0, 0, 0xD0, 0x07, 0, 0, 0, 0],
        )))
        .unwrap();
        let got = rx.recv(Some(Duration::from_secs(1))).unwrap().unwrap();
        assert_eq!(got.field("Gear"), Some(&Value::Str("P".into())));
        // An unknown identifier decodes as a nameless message.
        raw.send(&Message::raw(Packet::can(0x7FF, vec![1])))
            .unwrap();
        assert_eq!(
            rx.recv(Some(Duration::from_secs(1))).unwrap().unwrap().name,
            None
        );
        assert!(tx.send(&Message::new("Nope", vec![])).is_err());
    }

    #[test]
    fn periodic_follows_updates() {
        let tx = Channel::open("mem://t-ch-periodic", "raw", Options::new()).unwrap();
        let rx = Channel::open("mem://t-ch-periodic", "raw", Options::new()).unwrap();
        let task = tx
            .every(
                Duration::from_millis(10),
                Message::raw(Packet::bytes(vec![1])),
                None,
            )
            .unwrap();
        // Driven by received packets, not sleeps: a shared CI runner can
        // stall the sender thread.
        let wait = Some(Duration::from_secs(5));
        let first = rx.recv(wait).unwrap().unwrap();
        assert_eq!(first.packet.unwrap().data, vec![1]);
        task.update(Message::raw(Packet::bytes(vec![2])));
        let mut payloads = vec![vec![1]];
        while payloads.last() != Some(&vec![2]) {
            payloads.push(rx.recv(wait).unwrap().unwrap().packet.unwrap().data);
        }
        task.stop();
        while let Some(m) = rx.recv(Some(Duration::ZERO)).unwrap() {
            payloads.push(m.packet.unwrap().data);
        }
        assert_eq!(payloads.len() as u64, task.sent_count());
        let switch = payloads.iter().position(|p| p == &vec![2]).unwrap();
        assert!(payloads[..switch].iter().all(|p| p == &vec![1]));
        assert!(payloads[switch..].iter().all(|p| p == &vec![2]));
    }

    #[test]
    fn periodic_duration_ends() {
        let tx = Channel::open("mem://t-ch-duration", "raw", Options::new()).unwrap();
        let task = tx
            .every(
                Duration::from_millis(5),
                Message::raw(Packet::bytes(vec![])),
                Some(Duration::from_millis(30)),
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while task.is_running() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!task.is_running());
        let sent = task.sent_count();
        assert!((1..=7).contains(&sent), "sent {sent}");
    }

    #[test]
    fn subscriptions_serve_several_buses_at_once() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<(usize, Result<Message>)>();
        let buses: Vec<Channel> = (0..3)
            .map(|i| Channel::open(&format!("mem://t-sub-{i}"), "raw", Options::new()).unwrap())
            .collect();
        let subs: Vec<Subscription> = buses
            .iter()
            .enumerate()
            .map(|(i, bus)| {
                let tx = tx.clone();
                bus.subscribe(move |m| tx.send((i, m)).is_ok()).unwrap()
            })
            .collect();
        for i in (0..3).rev() {
            let sender = Channel::open(&format!("mem://t-sub-{i}"), "raw", Options::new()).unwrap();
            sender
                .send(&Message::raw(Packet::bytes(vec![i as u8])))
                .unwrap();
        }
        let mut got: Vec<(usize, Vec<u8>)> = (0..3)
            .map(|_| {
                let (i, m) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
                (i, m.unwrap().packet.unwrap().data)
            })
            .collect();
        got.sort();
        assert_eq!(got, vec![(0, vec![0]), (1, vec![1]), (2, vec![2])]);
        // Closing a bus ends its subscription with one Closed.
        buses[1].close();
        let (i, m) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(i == 1 && matches!(m, Err(Error::Closed)));
        let deadline = Instant::now() + Duration::from_secs(5);
        while subs[1].is_running() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!subs[1].is_running());
        subs[0].stop();
        assert!(!subs[0].is_running());
        assert!(buses[1].subscribe(|_| true).is_err());
    }

    #[test]
    fn persistent_receive_errors_back_off() {
        struct Failing;
        impl Transport for Failing {
            fn send(&self, _: &Packet) -> Result<()> {
                Ok(())
            }
            fn recv(&self, _: Option<Duration>) -> Result<Option<Packet>> {
                Err(Error::Decode("broken".into()))
            }
        }
        let ch = Channel::new(
            Arc::new(Failing),
            codec::open("raw", Options::new()).unwrap(),
        );
        let errors = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = Arc::clone(&errors);
        let sub = ch
            .subscribe(move |m| {
                assert!(m.is_err());
                count.fetch_add(1, Ordering::Relaxed);
                true
            })
            .unwrap();
        std::thread::sleep(Duration::from_millis(500));
        sub.stop();
        // 0 + 10 + 20 + 40 + 80 + 160 ms ... : about 7 errors in 500 ms,
        // not millions.
        let n = errors.load(Ordering::Relaxed);
        assert!((2..=12).contains(&n), "{n} errors");
    }

    #[test]
    fn close_and_errors() {
        let ch = Channel::open("mem://t-ch-close", "raw", Options::new()).unwrap();
        ch.close();
        assert!(matches!(ch.recv(Some(Duration::ZERO)), Err(Error::Closed)));
        assert!(matches!(
            ch.send(&Message::raw(Packet::bytes(vec![]))),
            Err(Error::Closed)
        ));
        assert!(Channel::open("nope://x", "raw", Options::new()).is_err());
        assert!(Channel::open("mem://x", "nope", Options::new()).is_err());
        let mut o = Options::new();
        o.insert("typo", Value::Bool(true));
        assert!(Channel::open("mem://x", "raw", o).is_err());
    }

    #[test]
    fn sbf_over_tcp() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let server = Channel::open(
            &format!("tcp://127.0.0.1:{port}?listen"),
            "sbf",
            Options::new(),
        )
        .unwrap();
        let client =
            Channel::open(&format!("tcp://127.0.0.1:{port}"), "sbf", Options::new()).unwrap();
        // The server sends to clients connected so far; retry until accepted.
        let msg = Message::new("AttEuler", vec![("heading".into(), Value::Float(42.0))]);
        let deadline = Instant::now() + Duration::from_secs(5);
        let got = loop {
            server.send(&msg).unwrap();
            if let Some(m) = client.recv(Some(Duration::from_millis(50))).unwrap() {
                break m;
            }
            assert!(Instant::now() < deadline, "no SBF block received");
        };
        assert_eq!(got.name.as_deref(), Some("AttEuler"));
        assert_eq!(got.field("heading"), Some(&Value::Float(42.0)));
    }
}
