//! `mem://name`: an in-process bus. Every channel opened on the same name
//! receives the packets the others send (CAN frames and byte packets alike),
//! with no kernel interface: for tests and for wiring simulated ECUs together.
//!
//! Options: `echo` (receive own packets), `filter=ID[:MASK],...` (CAN).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use super::can_frame::Filter;
use super::Transport;
use crate::error::Result;
use crate::message::{now_timestamp, Packet};
use crate::queue::Queue;
use crate::url::Endpoint;

struct Port {
    queue: Queue,
    filters: Vec<Filter>,
}

#[derive(Default)]
struct Hub {
    endpoints: Mutex<Vec<Weak<Port>>>,
}

fn hub(name: &str) -> Arc<Hub> {
    static HUBS: OnceLock<Mutex<HashMap<String, Weak<Hub>>>> = OnceLock::new();
    let mut hubs = HUBS.get_or_init(Default::default).lock().unwrap();
    hubs.retain(|_, h| h.strong_count() > 0);
    if let Some(h) = hubs.get(name).and_then(Weak::upgrade) {
        return h;
    }
    let h = Arc::new(Hub::default());
    hubs.insert(name.to_owned(), Arc::downgrade(&h));
    h
}

pub(crate) struct Mem {
    hub: Arc<Hub>,
    me: Arc<Port>,
    echo: bool,
}

impl Mem {
    pub(crate) fn open(endpoint: Endpoint) -> Result<Self> {
        let mut o = endpoint.options;
        let echo = o.take_bool("echo", false)?;
        let filters = match o.take_str("filter")? {
            Some(spec) => Filter::parse_list(&spec)?,
            None => Vec::new(),
        };
        o.finish("mem://")?;
        let hub = hub(&endpoint.address);
        let me = Arc::new(Port {
            queue: Queue::default(),
            filters,
        });
        hub.endpoints.lock().unwrap().push(Arc::downgrade(&me));
        Ok(Self { hub, me, echo })
    }
}

impl Transport for Mem {
    fn send(&self, packet: &Packet) -> Result<()> {
        if packet.can.is_some() {
            super::can_frame::check(packet, true)?;
        }
        let mut packet = packet.clone();
        packet.timestamp = Some(now_timestamp());
        let receivers: Vec<Arc<Port>> = {
            let mut eps = self.hub.endpoints.lock().unwrap();
            eps.retain(|e| e.strong_count() > 0);
            eps.iter().filter_map(Weak::upgrade).collect()
        };
        for ep in receivers {
            if (!Arc::ptr_eq(&ep, &self.me) || self.echo) && Filter::accepts(&ep.filters, &packet) {
                ep.queue.push(packet.clone());
            }
        }
        Ok(())
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        self.me.queue.pop(timeout)
    }

    fn close(&self) {
        self.me.queue.close();
        self.me.queue.retain(|_| false);
    }
}

#[cfg(test)]
mod tests {
    use crate::message::Packet;
    use crate::transport::open;
    use std::time::Duration;

    #[test]
    fn delivers_to_other_endpoints() {
        let a = open("mem://t-mem").unwrap();
        let b = open("mem://t-mem").unwrap();
        let other = open("mem://t-mem-other").unwrap();
        a.send(&Packet::can(0x123, vec![1, 2])).unwrap();
        a.send(&Packet::bytes(b"xyz".to_vec())).unwrap();
        let got = b.recv(Some(Duration::from_millis(100))).unwrap().unwrap();
        assert_eq!(got.can.unwrap().id, 0x123);
        assert!(got.timestamp.is_some());
        assert_eq!(b.recv(Some(Duration::ZERO)).unwrap().unwrap().data, b"xyz");
        assert!(a.recv(Some(Duration::ZERO)).unwrap().is_none());
        assert!(other.recv(Some(Duration::ZERO)).unwrap().is_none());
    }

    #[test]
    fn echo_and_filters() {
        let a = open("mem://t-mem-f?echo&filter=0x100:0x700").unwrap();
        a.send(&Packet::can(0x1AB, vec![])).unwrap();
        a.send(&Packet::can(0x2AB, vec![])).unwrap();
        assert_eq!(
            a.recv(Some(Duration::ZERO))
                .unwrap()
                .unwrap()
                .can
                .unwrap()
                .id,
            0x1AB
        );
        assert!(a.recv(Some(Duration::ZERO)).unwrap().is_none());
        assert!(crate::transport::open("mem://x?nope").is_err());
    }

    #[test]
    fn close_wakes_a_receiver() {
        let a = std::sync::Arc::new(open("mem://t-mem-close").unwrap());
        let b = std::sync::Arc::clone(&a);
        let h = std::thread::spawn(move || b.recv(None));
        std::thread::sleep(Duration::from_millis(30));
        a.close();
        assert!(h.join().unwrap().is_err());
    }
}
