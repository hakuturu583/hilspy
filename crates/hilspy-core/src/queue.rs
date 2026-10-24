//! A blocking queue of received packets, shared by the transports that
//! receive on background threads (mem, tcp); an empty one is a receive that
//! waits for its timeout or `close` (v4l2).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::message::Packet;

/// Packets a queue holds before it drops the oldest.
const LIMIT: usize = 1 << 16;

#[derive(Default)]
pub(crate) struct Queue {
    state: Mutex<State>,
    ready: Condvar,
}

#[derive(Default)]
struct State {
    packets: VecDeque<Packet>,
    closed: bool,
}

impl Queue {
    pub(crate) fn push(&self, packet: Packet) {
        let mut s = self.state.lock().unwrap();
        if s.closed {
            return;
        }
        if s.packets.len() >= LIMIT {
            s.packets.pop_front();
        }
        s.packets.push_back(packet);
        self.ready.notify_one();
    }

    /// The next packet; `None` on timeout; `Closed` once closed and drained.
    pub(crate) fn pop(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(p) = s.packets.pop_front() {
                return Ok(Some(p));
            }
            if s.closed {
                return Err(Error::Closed);
            }
            s = match deadline {
                None => self.ready.wait(s).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Ok(None);
                    }
                    self.ready.wait_timeout(s, d - now).unwrap().0
                }
            };
        }
    }

    /// Wake receivers; packets already queued can still be read.
    pub(crate) fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.ready.notify_all();
    }

    pub(crate) fn retain(&self, keep: impl Fn(&Packet) -> bool) {
        self.state.lock().unwrap().packets.retain(keep);
    }
}
