//! An in-process bus: every endpoint opened on a channel name receives the
//! frames the other endpoints send.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

use super::bus::{BusBackend, BusConfig};
use super::filter::Filter;
use super::frame::{now_timestamp, Frame};
use crate::error::{Error, Result};

/// Frames an endpoint queues before it drops the oldest.
const QUEUE_LIMIT: usize = 1 << 16;

struct Endpoint {
    queue: Mutex<EndpointQueue>,
    ready: Condvar,
    filters: RwLock<Vec<Filter>>,
    fd: bool,
}

struct EndpointQueue {
    frames: VecDeque<Frame>,
    closed: bool,
}

impl Endpoint {
    fn deliver(&self, frame: &Frame) {
        if frame.fd && !self.fd {
            // A classic controller does not see CAN FD frames.
            return;
        }
        if !Filter::accepts(&self.filters.read().unwrap(), frame) {
            return;
        }
        let mut queue = self.queue.lock().unwrap();
        if queue.closed {
            return;
        }
        if queue.frames.len() >= QUEUE_LIMIT {
            queue.frames.pop_front();
        }
        queue.frames.push_back(frame.clone());
        self.ready.notify_one();
    }
}

#[derive(Default)]
struct Hub {
    endpoints: Mutex<Vec<Weak<Endpoint>>>,
}

fn hub(channel: &str) -> Arc<Hub> {
    static HUBS: OnceLock<Mutex<HashMap<String, Weak<Hub>>>> = OnceLock::new();
    let mut hubs = HUBS.get_or_init(Default::default).lock().unwrap();
    hubs.retain(|_, h| h.strong_count() > 0);
    if let Some(h) = hubs.get(channel).and_then(Weak::upgrade) {
        return h;
    }
    let h = Arc::new(Hub::default());
    hubs.insert(channel.to_owned(), Arc::downgrade(&h));
    h
}

pub(crate) struct VirtualBus {
    hub: Arc<Hub>,
    endpoint: Arc<Endpoint>,
    receive_own_messages: bool,
}

impl VirtualBus {
    pub(crate) fn open(config: &BusConfig) -> Result<Self> {
        let hub = hub(&config.channel);
        let endpoint = Arc::new(Endpoint {
            queue: Mutex::new(EndpointQueue {
                frames: VecDeque::new(),
                closed: false,
            }),
            ready: Condvar::new(),
            filters: RwLock::new(config.filters.clone()),
            fd: config.fd,
        });
        hub.endpoints
            .lock()
            .unwrap()
            .push(Arc::downgrade(&endpoint));
        Ok(Self {
            hub,
            endpoint,
            receive_own_messages: config.receive_own_messages,
        })
    }
}

impl BusBackend for VirtualBus {
    fn send(&self, frame: &Frame) -> Result<()> {
        let mut frame = frame.clone();
        frame.timestamp = Some(now_timestamp());
        let receivers: Vec<Arc<Endpoint>> = {
            let mut endpoints = self.hub.endpoints.lock().unwrap();
            endpoints.retain(|e| e.strong_count() > 0);
            endpoints.iter().filter_map(Weak::upgrade).collect()
        };
        for endpoint in receivers {
            if !Arc::ptr_eq(&endpoint, &self.endpoint) || self.receive_own_messages {
                endpoint.deliver(&frame);
            }
        }
        Ok(())
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Frame>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut queue = self.endpoint.queue.lock().unwrap();
        loop {
            if queue.closed {
                return Err(Error::BusClosed);
            }
            if let Some(frame) = queue.frames.pop_front() {
                return Ok(Some(frame));
            }
            queue = match deadline {
                None => self.endpoint.ready.wait(queue).unwrap(),
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Ok(None);
                    }
                    self.endpoint
                        .ready
                        .wait_timeout(queue, deadline - now)
                        .unwrap()
                        .0
                }
            };
        }
    }

    fn set_filters(&self, filters: &[Filter]) -> Result<()> {
        *self.endpoint.filters.write().unwrap() = filters.to_vec();
        // Frames already queued were accepted by the old filters.
        let mut queue = self.endpoint.queue.lock().unwrap();
        queue.frames.retain(|f| Filter::accepts(filters, f));
        Ok(())
    }

    fn shutdown(&self) {
        let mut queue = self.endpoint.queue.lock().unwrap();
        queue.closed = true;
        queue.frames.clear();
        self.endpoint.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Bus, BusConfig, Filter, Frame};
    use std::time::Duration;

    fn open(channel: &str) -> Bus {
        Bus::open(BusConfig::new("virtual", channel)).unwrap()
    }

    #[test]
    fn frames_reach_other_endpoints_only() {
        let a = open("t-basic");
        let b = open("t-basic");
        let other = open("t-basic-other");
        let frame = Frame::new(0x123, &[1, 2, 3], false).unwrap();
        a.send(&frame).unwrap();
        let got = b.recv(Some(Duration::from_millis(100))).unwrap().unwrap();
        assert_eq!(got.id, 0x123);
        assert_eq!(got.data, vec![1, 2, 3]);
        assert!(got.timestamp.is_some());
        assert!(a.recv(Some(Duration::ZERO)).unwrap().is_none());
        assert!(other.recv(Some(Duration::ZERO)).unwrap().is_none());
    }

    #[test]
    fn filters_and_own_messages() {
        let mut config = BusConfig::new("virtual", "t-filter");
        config.receive_own_messages = true;
        config.filters = vec![Filter {
            id: 0x100,
            mask: 0x700,
            extended: Some(false),
        }];
        let a = Bus::open(config).unwrap();
        a.send(&Frame::new(0x1AB, &[], false).unwrap()).unwrap();
        a.send(&Frame::new(0x2AB, &[], false).unwrap()).unwrap();
        a.send(&Frame::new(0x1AB, &[], true).unwrap()).unwrap();
        let got = a.recv(Some(Duration::ZERO)).unwrap().unwrap();
        assert_eq!((got.id, got.extended), (0x1AB, false));
        assert!(a.recv(Some(Duration::ZERO)).unwrap().is_none());
    }

    #[test]
    fn fd_frames_need_an_fd_bus() {
        let classic = open("t-fd");
        let mut config = BusConfig::new("virtual", "t-fd");
        config.fd = true;
        let fd = Bus::open(config).unwrap();
        let frame = Frame::new_fd(0x10, &[0; 16], false, true).unwrap();
        assert!(classic.send(&frame).is_err());
        fd.send(&frame).unwrap();
        assert!(classic.recv(Some(Duration::ZERO)).unwrap().is_none());
    }

    #[test]
    fn shutdown_wakes_a_blocked_receiver() {
        let a = open("t-shutdown");
        let b = a.clone();
        let handle = std::thread::spawn(move || b.recv(None));
        std::thread::sleep(Duration::from_millis(50));
        a.shutdown();
        assert!(handle.join().unwrap().is_err());
    }
}
