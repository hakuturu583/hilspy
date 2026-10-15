//! A channel: a transport and a codec composed, sending and receiving named
//! messages.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::codec::{self, Codec};
use crate::error::{Error, Result};
use crate::message::Message;
use crate::transport::{self, Transport};
use crate::value::Options;

struct Inner {
    transport: Arc<dyn Transport>,
    codec: Mutex<Box<dyn Codec>>,
    pending: Mutex<VecDeque<Message>>,
    closed: AtomicBool,
}

impl Inner {
    fn send(&self, message: &Message) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let packets = match (&message.name, &message.packet) {
            (None, Some(p)) => vec![p.clone()],
            (None, None) => {
                return Err(Error::encode("a message needs a name or data"));
            }
            (Some(_), _) => self.codec.lock().unwrap().encode(message)?,
        };
        for p in &packets {
            self.transport.send(p)?;
        }
        Ok(())
    }
}

/// A transport and a codec composed. Cloning shares the channel.
#[derive(Clone)]
pub struct Channel {
    inner: Arc<Inner>,
}

impl Channel {
    /// Open the transport `url` with the codec `codec` (`raw`, `sbf`,
    /// `dbc:FILE`, ...) configured by `options`.
    pub fn open(url: &str, codec: &str, options: Options) -> Result<Self> {
        let codec = codec::open(codec, options)?;
        let transport = transport::open(url)?;
        Ok(Self::new(transport, codec))
    }

    pub fn new(transport: Arc<dyn Transport>, codec: Box<dyn Codec>) -> Self {
        Self {
            inner: Arc::new(Inner {
                transport,
                codec: Mutex::new(codec),
                pending: Mutex::new(VecDeque::new()),
                closed: AtomicBool::new(false),
            }),
        }
    }

    /// Encode and send a message (a nameless one sends its packet as is).
    pub fn send(&self, message: &Message) -> Result<()> {
        self.inner.send(message)
    }

    /// The next decoded message, or `None` once `timeout` passes.
    pub fn recv(&self, timeout: Option<Duration>) -> Result<Option<Message>> {
        let inner = &self.inner;
        if let Some(m) = inner.pending.lock().unwrap().pop_front() {
            return Ok(Some(m));
        }
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if inner.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            let Some(packet) = inner.transport.recv(remaining)? else {
                return Ok(None);
            };
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
    /// time-stamping codecs stamp each one.
    pub fn every(
        &self,
        period: Duration,
        message: Message,
        duration: Option<Duration>,
    ) -> Result<Periodic> {
        Periodic::start(Arc::clone(&self.inner), period, message, duration)
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
    thread: Option<JoinHandle<()>>,
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
        // Fail now, not on the first tick, on a message that cannot encode.
        if message.name.is_some() {
            inner.codec.lock().unwrap().encode(&message)?;
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                message,
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
            .spawn(move || run(inner, worker, duration))?;
        Ok(Self {
            shared,
            thread: Some(thread),
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
            && self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Stop and wait for the sender thread (idempotent).
    pub fn stop(&mut self) {
        self.shared.state.lock().unwrap().stopped = true;
        self.shared.wake.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Periodic {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(inner: Arc<Inner>, shared: Arc<Shared>, duration: Option<Duration>) {
    let start = Instant::now();
    let end = duration.map(|d| start + d);
    let mut next = start;
    let mut state = shared.state.lock().unwrap();
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
        let mut task = tx
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
