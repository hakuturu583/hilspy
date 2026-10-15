//! ASAM FIBEX (MCD-2 NET) network descriptions: the CAN / CAN FD or LIN
//! frames of one channel, converted to a DBC [`Database`] so the `dbc`
//! encoding, decoding and E2E protection apply unchanged, plus what the
//! channel's bus adds ([`Bus`]: LIN identifiers, checksum models and schedule
//! tables).
//!
//! The bus is the `PROTOCOL` of the cluster whose `CHANNEL-REFS` name the
//! channel: `LIN` makes a LIN channel, anything else (or no cluster) CAN.
//!
//! LIN: a frame triggering's `IDENTIFIER-VALUE` is the frame identifier
//! (0..=63). Its checksum model is the triggering's `CHECKSUM-TYPE` (or
//! `LIN-CHECKSUM`) `CLASSIC` / `ENHANCED` if it has one, else classic for a
//! cluster `PROTOCOL-VERSION` 1.x and enhanced otherwise; the diagnostic
//! frames `0x3C` / `0x3D` are always classic. Schedule tables are the
//! `SCHEDULE-TABLE`s of the channel (or its cluster): each entry with a
//! `DELAY` (an `xs:duration`, the time from the entry's start to the next
//! one's) and, for frame entries, a `FRAME-TRIGGERING-REF`. A table's period
//! is the sum of its delays; a frame's cycle time is the period divided by
//! the number of the table's entries that send it (exact when they are
//! evenly spaced, the mean interval otherwise), from the table chosen with
//! `schedule`, else the first table (in document order) that sends the
//! frame, else its triggering's `CYCLIC-TIMING`.
//!
//! The `afibex` crate parses the file: channels, frames with their PDU
//! instances, PDUs with their signal instances, signals and the coded types
//! of codings. What it leaves out is read from the same document with
//! `roxmltree`, by element ID: the frame triggerings of a channel (CAN
//! identifier, extended addressing, CAN FD, the `CYCLIC-TIMING` that becomes
//! the message's cycle time), the connectors of the ECUs (the `OUTPUT-PORT`s
//! and `INPUT-PORT`s that refer to a frame triggering make its message's
//! transmitters and receivers; the ECUs are the database's nodes) and the compu methods of a coding
//! (scales with their limits, rational coefficients and texttable names,
//! physical constraints, units).
//!
//! Multiplexed PDUs: the `SWITCH` becomes a multiplexor signal, each switched
//! PDU's signals are multiplexed on its `SWITCH-CODE`s (consecutive codes of
//! one PDU merged into ranges, like `SG_MUL_VAL_`), and the `STATIC-PART`'s
//! signals are plain. `afibex` reads the switch and the dynamic part; the
//! static part, which it skips, comes from `roxmltree`. Switched and static
//! PDUs are laid out in the multiplexed PDU's bit numbering (AUTOSAR copies
//! segments at the same positions; `canmatrix` writes them so), so their
//! instances must be at bit 0, low-high.
//!
//! Coded types: unsigned and two's complement integers map onto the DBC
//! value types; sign-magnitude (`SM`) and one's complement (`1C`) integers
//! onto [`ValueKind::SignMagnitude`] and [`ValueKind::OnesComplement`].
//!
//! Compu methods: `IDENTICAL`, `LINEAR`, `SCALE-LINEAR`, `TEXTTABLE`,
//! `RAT-FUNC` and `SCALE-RAT-FUNC` (not `FORMULA`). A coding whose scales are
//! one linear function plus texttable entries for single integers becomes the
//! signal's factor, offset and value table, as in a DBC; anything else
//! (texttable ranges, piecewise or rational functions) becomes the signal's
//! [`Coding`], which the `dbc` encoding and decoding consult instead.
//!
//! Layout: a signal instance's `BIT-POSITION` is read as a DBC start bit
//! within its PDU (bit 0 is the least significant bit of byte 0): the least
//! significant bit for `IS-HIGH-LOW-BYTE-ORDER` false (Intel, the default),
//! the most significant bit for true (Motorola), as `canmatrix` reads and
//! writes FIBEX. A PDU instance's `BIT-POSITION` places the PDU's bits in the
//! frame: for a low-high instance (`IS-HIGH-LOW-BYTE-ORDER` false, the
//! default) PDU bit `b` (numbered like a start bit) is frame bit
//! `offset + b`; a high-low instance holds the PDU big-endian, as a Motorola
//! field of the PDU's length whose most significant bit is at the position,
//! so PDU bit `b` is value bit `b` of that field and the PDU's last byte comes
//! first. A signal's frame bits are those of its PDU bits (`Placement`); they
//! must again form one DBC signal, which they always do at a byte boundary (a
//! high-low instance turns Intel signals into Motorola ones and back). Off a
//! byte boundary an Intel signal still does, while a Motorola signal that
//! crosses a byte of its PDU is split in the frame and rejected.

use std::collections::HashMap;
use std::ops::Bound;
use std::path::Path;
use std::time::Duration;

use afibex::fibex::{
    BaseDataType, Category, Encoding, FibexData, Pdu, PduInstance, SignalInstance,
};
use roxmltree::{Document, Node};

use crate::dbc::{
    bits, ByteOrder, Coding, Conversion, Database, Message, Multiplexing, Scale, Signal, ValueKind,
};
use crate::error::{Error, Result};
use crate::message::LinChecksum;

fn bad(msg: impl Into<String>) -> Error {
    Error::FibexParse(msg.into())
}

/// The frames of one channel of a FIBEX file as a database. `channel` is a
/// channel short name or ID; it may be left out when the file has a single
/// channel. See [`load_channel`] for what a LIN channel adds.
pub fn load(path: impl AsRef<Path>, channel: Option<&str>) -> Result<Database> {
    Ok(load_channel(path, channel, None)?.database)
}

/// One channel of a FIBEX file: its frames as a database, and its bus.
#[derive(Clone, Debug)]
pub struct ChannelDescription {
    pub database: Database,
    pub bus: Bus,
}

/// What a channel's bus adds to its frames.
#[derive(Clone, Debug, PartialEq)]
pub enum Bus {
    /// CAN / CAN FD: frames by the database's `id` and `extended`.
    Can,
    /// LIN: the database's `id` is the frame identifier.
    Lin(Lin),
}

/// The LIN side of a channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Lin {
    /// The channel's frames, in triggering order.
    pub frames: Vec<LinFrame>,
    /// The channel's schedule tables, in document order.
    pub schedule_tables: Vec<ScheduleTable>,
}

/// A LIN frame of a channel.
#[derive(Clone, Debug, PartialEq)]
pub struct LinFrame {
    /// The database message's name.
    pub name: String,
    /// Frame identifier, 0..=63.
    pub id: u8,
    pub checksum: LinChecksum,
}

/// A LIN schedule table.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleTable {
    /// `SHORT-NAME`, else `ID`.
    pub name: String,
    pub entries: Vec<ScheduleEntry>,
}

/// One slot of a schedule table.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleEntry {
    /// The frame it sends (the database message's name); `None` for other
    /// entries (diagnostic or free-format commands), which still take their
    /// delay.
    pub frame: Option<String>,
    /// From the entry's start to the next entry's.
    pub delay: Duration,
}

impl ScheduleTable {
    /// One pass through the table: the sum of its delays.
    pub fn period(&self) -> Duration {
        self.entries.iter().map(|e| e.delay).sum()
    }

    /// How often the table sends `frame`: its period divided by the number
    /// of entries that send it; `None` if none do or the period is zero.
    pub fn cycle_time(&self, frame: &str) -> Option<Duration> {
        let n = self
            .entries
            .iter()
            .filter(|e| e.frame.as_deref() == Some(frame))
            .count();
        let period = self.period();
        (n > 0 && !period.is_zero()).then(|| period / n as u32)
    }

    /// Whether any entry sends `frame`.
    pub fn sends(&self, frame: &str) -> bool {
        self.entries
            .iter()
            .any(|e| e.frame.as_deref() == Some(frame))
    }
}

/// One channel of a FIBEX file. `channel` is a channel short name or ID (it
/// may be left out when the file has a single channel); `schedule` names the
/// LIN schedule table (short name or ID) that gives a LIN channel's frames
/// their cycle times, and is an error on other channels.
pub fn load_channel(
    path: impl AsRef<Path>,
    channel: Option<&str>,
    schedule: Option<&str>,
) -> Result<ChannelDescription> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })?;
    let xml = Document::parse(&text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let root = xml.root_element().tag_name().name();
    if root != "FIBEX" {
        return Err(bad(format!(
            "{}: the root element is {root}, not FIBEX",
            path.display()
        )));
    }
    let mut data = FibexData::new();
    data.load_fibex_file(path)
        .map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let ids = xml
        .descendants()
        .filter_map(|n| Some((attr(n, "ID")?, n)))
        .collect();
    let (ecus, ports) = ecus(&xml);
    Network {
        data: &data,
        ids,
        ecus,
        ports,
        xml: &xml,
    }
    .channel(channel, schedule)
}

/// The ECUs that send and receive a frame triggering, by name.
#[derive(Default)]
struct Ports {
    outputs: Vec<String>,
    inputs: Vec<String>,
}

/// The ECUs of the file by name (`SHORT-NAME`, else `ID`), and the ports of
/// their connectors by frame triggering ID (`OUTPUT-PORT` /
/// `INPUT-PORT` `FRAME-TRIGGERING-REF`s). `afibex` skips connectors.
fn ecus<'a>(xml: &'a Document<'_>) -> (Vec<String>, HashMap<&'a str, Ports>) {
    let mut names = Vec::new();
    let mut ports: HashMap<&str, Ports> = HashMap::new();
    let ecu_elements = xml.descendants().filter(|n| {
        n.is_element()
            && n.tag_name().name() == "ECU"
            && n.parent_element()
                .is_some_and(|p| p.tag_name().name() == "ECUS")
    });
    for ecu in ecu_elements {
        let Some(name) = child(ecu, "SHORT-NAME")
            .map(text)
            .filter(|n| !n.is_empty())
            .or_else(|| attr(ecu, "ID"))
        else {
            continue;
        };
        if !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
        for port in ecu.descendants().filter(|n| n.is_element()) {
            let output = match port.tag_name().name() {
                "OUTPUT-PORT" => true,
                "INPUT-PORT" => false,
                _ => continue,
            };
            for t in children(port, "FRAME-TRIGGERING-REF").filter_map(|r| attr(r, "ID-REF")) {
                let p = ports.entry(t).or_default();
                let list = if output {
                    &mut p.outputs
                } else {
                    &mut p.inputs
                };
                if !list.iter().any(|n| n == name) {
                    list.push(name.to_owned());
                }
            }
        }
    }
    (names, ports)
}

/// The value of an attribute by local name (any namespace).
fn attr<'a>(n: Node<'a, '_>, name: &str) -> Option<&'a str> {
    n.attributes().find(|a| a.name() == name).map(|a| a.value())
}

fn children<'a, 'i>(n: Node<'a, 'i>, name: &'a str) -> impl Iterator<Item = Node<'a, 'i>> + 'a
where
    'i: 'a,
{
    n.children()
        .filter(move |c| c.is_element() && c.tag_name().name() == name)
}

fn child<'a, 'i>(n: Node<'a, 'i>, name: &'a str) -> Option<Node<'a, 'i>>
where
    'i: 'a,
{
    children(n, name).next()
}

fn text<'a>(n: Node<'a, '_>) -> &'a str {
    n.text().unwrap_or_default().trim()
}

fn number(n: Node<'_, '_>, what: &str) -> Result<f64> {
    text(n)
        .parse()
        .map_err(|_| bad(format!("{what}: {:?} is not a number", text(n))))
}

/// A frame triggering: how a channel carries a frame.
struct Triggering {
    /// The triggering's own `ID`.
    tid: String,
    id: u32,
    extended: bool,
    fd: bool,
    frame: String,
    /// The `REPEATING-TIME-RANGE` of its first `CYCLIC-TIMING`.
    cycle_time: Option<Duration>,
}

/// The physical meaning of a coding: its scales, range and unit.
#[derive(Default)]
struct Compu {
    scales: Vec<Scale>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    unit: String,
}

/// What the DBC model's own fields express of a coding's scales.
struct Plain {
    factor: f64,
    offset: f64,
    choices: Vec<(i64, String)>,
}

impl Compu {
    /// The scales as one factor and offset plus exact value table entries,
    /// when they are that: every name for a single integer and every
    /// rational scale the same linear function (whatever its limits).
    fn plain(&self) -> Option<Plain> {
        let mut linear = None;
        let mut choices = Vec::new();
        for scale in &self.scales {
            match &scale.conversion {
                Conversion::Text(name) => match (scale.lower, scale.upper) {
                    (Bound::Included(l), Bound::Included(u))
                        if l == u && l.fract() == 0.0 && l.abs() < 9e15 =>
                    {
                        choices.push((l as i64, name.clone()));
                    }
                    _ => return None,
                },
                Conversion::Rational {
                    numerator,
                    denominator,
                } => {
                    let (offset, factor) = match (numerator.as_slice(), denominator.as_slice()) {
                        ([offset, factor], [d]) => (offset / d, factor / d),
                        _ => return None,
                    };
                    if factor == 0.0 || !factor.is_finite() || !offset.is_finite() {
                        return None;
                    }
                    match linear {
                        Some(l) if l != (factor, offset) => return None,
                        _ => linear = Some((factor, offset)),
                    }
                }
            }
        }
        choices.sort_by_key(|(v, _)| *v);
        let (factor, offset) = linear.unwrap_or((1.0, 0.0));
        Some(Plain {
            factor,
            offset,
            choices,
        })
    }
}

/// Where the bits of a PDU instance sit in its frame.
struct Placement {
    /// The frame `(byte, bit)` of each PDU bit, by PDU bit number (a DBC
    /// start bit: byte * 8 + bit, bit 0 the least significant).
    bits: Vec<(usize, u32)>,
    frame_len: usize,
    /// For errors: the frame's name and the instance's `BIT-POSITION`.
    frame: String,
    offset: u32,
}

impl Placement {
    /// A PDU of `len` bytes at the instance's `BIT-POSITION` in a frame of
    /// `frame_len` bytes; `None` if it does not fit. Low-high (Intel): the
    /// position is the PDU's least significant bit and PDU bit `b` is frame
    /// bit `offset + b`. High-low: the PDU is a Motorola field whose most
    /// significant bit is at the position, so its last byte comes first.
    fn new(offset: u32, high_low: bool, len: usize, frame_len: usize, frame: &str) -> Option<Self> {
        let size = u32::try_from(len * 8).ok()?;
        let bits = if high_low {
            bits::positions(offset, size, ByteOrder::BigEndian, frame_len)?
        } else {
            if u64::from(offset) + u64::from(size) > frame_len as u64 * 8 {
                return None;
            }
            (offset..offset + size)
                .map(|b| ((b / 8) as usize, b % 8))
                .collect()
        };
        Some(Self {
            bits,
            frame_len,
            frame: frame.to_owned(),
            offset,
        })
    }

    /// Move signal `name` of PDU `pdu`, positioned in the placed PDU of `len`
    /// bytes, to the frame; an error if its bits are not one DBC signal there.
    fn apply(&self, signal: &mut Signal, len: usize, pdu: &str, name: &str) -> Result<()> {
        let layout = bits::positions(signal.start_bit, signal.size, signal.byte_order, len)
            .and_then(|in_pdu| {
                let in_frame: Vec<_> = in_pdu
                    .into_iter()
                    .map(|(byte, bit)| self.bits[byte * 8 + bit as usize])
                    .collect();
                bits::layout(&in_frame, signal.byte_order, self.frame_len)
            });
        let (start, order) = layout.ok_or_else(|| {
            bad(format!(
                "signal {name} of PDU {pdu}: its bits are not one signal in frame {} (a signal \
                 that crosses a byte of a PDU placed at bit {}, not at a byte boundary)",
                self.frame, self.offset
            ))
        })?;
        signal.start_bit = start;
        signal.byte_order = order;
        Ok(())
    }
}

struct Network<'d, 'a, 'i> {
    data: &'d FibexData,
    /// Elements by `ID` attribute.
    ids: HashMap<&'a str, Node<'a, 'i>>,
    /// ECU names, in document order.
    ecus: Vec<String>,
    /// The sending and receiving ECUs by frame triggering ID.
    ports: HashMap<&'a str, Ports>,
    xml: &'a Document<'i>,
}

/// The bus of a channel's cluster.
#[derive(Clone, Copy, PartialEq)]
enum BusKind {
    Can,
    Lin,
}

impl BusKind {
    /// The longest frame, in bytes.
    fn max_size(self) -> usize {
        match self {
            Self::Can => 64,
            Self::Lin => 8,
        }
    }
}

impl Network<'_, '_, '_> {
    fn channel(&self, channel: Option<&str>, schedule: Option<&str>) -> Result<ChannelDescription> {
        let channel = self.select(channel)?;
        let node = self
            .ids
            .get(channel)
            .copied()
            .ok_or_else(|| bad(format!("channel {channel} not found")))?;
        let cluster = self.cluster(channel);
        let protocol = cluster
            .and_then(|c| child(c, "PROTOCOL"))
            .map(|p| text(p).to_ascii_uppercase())
            .unwrap_or_default();
        let kind = if protocol.contains("LIN") {
            BusKind::Lin
        } else {
            BusKind::Can
        };
        if kind != BusKind::Lin && schedule.is_some() {
            return Err(Error::Config(format!(
                "option schedule: channel {channel} is not a LIN channel"
            )));
        }
        let triggerings = node
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "FRAME-TRIGGERING");
        let mut messages = Vec::new();
        let bus = match kind {
            BusKind::Can => {
                for t in triggerings {
                    if let Some(t) = triggering(t)? {
                        messages.push(self.message(&t, kind)?);
                    }
                }
                Bus::Can
            }
            BusKind::Lin => {
                let classic = cluster
                    .and_then(|c| child(c, "PROTOCOL-VERSION"))
                    .is_some_and(|v| text(v).starts_with('1'));
                // Frame names by triggering ID, for the schedule tables.
                let mut by_tid = HashMap::new();
                let mut frames = Vec::new();
                for t in triggerings {
                    let Some((t, id, checksum)) = lin_triggering(t, classic)? else {
                        continue;
                    };
                    let m = self.message(&t, kind)?;
                    by_tid.insert(t.tid.clone(), m.name.clone());
                    frames.push(LinFrame {
                        name: m.name.clone(),
                        id,
                        checksum,
                    });
                    messages.push(m);
                }
                let tables = schedule_tables(node, cluster, &by_tid)?;
                let chosen = match schedule {
                    None => None,
                    Some(s) => Some(
                        tables
                            .iter()
                            .find(|(id, t)| t.name == s || id == s)
                            .map(|(_, t)| t)
                            .ok_or_else(|| {
                                let names: Vec<_> =
                                    tables.iter().map(|(_, t)| t.name.as_str()).collect();
                                Error::Config(format!(
                                    "option schedule: no schedule table {s:?} on channel \
                                     {channel} (tables: {})",
                                    if names.is_empty() {
                                        "none".to_owned()
                                    } else {
                                        names.join(", ")
                                    }
                                ))
                            })?,
                    ),
                };
                for m in &mut messages {
                    let table =
                        chosen.or_else(|| tables.iter().map(|(_, t)| t).find(|t| t.sends(&m.name)));
                    if let Some(table) = table {
                        m.cycle_time = table.cycle_time(&m.name);
                    }
                }
                Bus::Lin(Lin {
                    frames,
                    schedule_tables: tables.into_iter().map(|(_, t)| t).collect(),
                })
            }
        };
        let database =
            Database::new(String::new(), self.ecus.clone(), messages).map_err(|e| match e {
                Error::DbcParse(m) => bad(format!("channel {channel}: {m}")),
                e => e,
            })?;
        Ok(ChannelDescription { database, bus })
    }

    /// The cluster whose `CHANNEL-REFS` name the channel.
    fn cluster(&self, channel: &str) -> Option<Node<'_, '_>> {
        self.xml.descendants().find(|n| {
            n.is_element()
                && n.tag_name().name() == "CLUSTER"
                && children(*n, "CHANNEL-REFS")
                    .flat_map(|r| children(r, "CHANNEL-REF"))
                    .any(|r| attr(r, "ID-REF") == Some(channel))
        })
    }

    /// The ID of the selected channel.
    fn select(&self, wanted: Option<&str>) -> Result<&str> {
        let mut channels: Vec<_> = self.data.elements.channels.values().collect();
        channels.sort_by(|a, b| a.id.cmp(&b.id));
        let list = || {
            channels
                .iter()
                .map(|c| match &c.short_name {
                    Some(n) => format!("{n} ({})", c.id),
                    None => c.id.clone(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        match wanted {
            Some(w) => channels
                .iter()
                .find(|c| c.id == w)
                .or_else(|| channels.iter().find(|c| c.short_name.as_deref() == Some(w)))
                .map(|c| c.id.as_str())
                .ok_or_else(|| {
                    Error::Config(format!(
                        "option channel: no FIBEX channel {w:?} (channels: {})",
                        list()
                    ))
                }),
            None => match channels.as_slice() {
                [c] => Ok(c.id.as_str()),
                [] => Err(Error::Config("the FIBEX file has no channels".into())),
                _ => Err(Error::Config(format!(
                    "the FIBEX file has {} channels; choose one with the channel option \
                     (channels: {})",
                    channels.len(),
                    list()
                ))),
            },
        }
    }

    fn message(&self, t: &Triggering, kind: BusKind) -> Result<Message> {
        let frame = self
            .data
            .elements
            .frames_map_by_id
            .get(&t.frame)
            .ok_or_else(|| {
                bad(format!(
                    "frame triggering refers to unknown frame {}",
                    t.frame
                ))
            })?;
        let name = frame.short_name.clone().unwrap_or_else(|| frame.id.clone());
        let size = frame.byte_length as usize;
        if size > kind.max_size() {
            return Err(bad(match kind {
                BusKind::Can => format!("frame {name} is {size} bytes (at most 64)"),
                BusKind::Lin => format!("LIN frame {name} is {size} bytes (at most 8)"),
            }));
        }
        // (PDU name, signal name, signal) for every signal of every PDU.
        let mut signals = Vec::new();
        for instance in &frame.pdu_instances {
            let pdu = self
                .data
                .elements
                .pdus_map_by_id
                .get(&instance.pdu_ref)
                .ok_or_else(|| {
                    bad(format!(
                        "frame {name} refers to unknown PDU {}",
                        instance.pdu_ref
                    ))
                })?;
            let pdu_name = pdu
                .short_name
                .clone()
                .unwrap_or_else(|| instance.pdu_ref.clone());
            let offset = instance.bit_position.unwrap_or(0);
            let len = pdu.byte_length as usize;
            let high_low = instance.is_high_low_byte_order == Some(true);
            let place = Placement::new(offset, high_low, len, size, &name).ok_or_else(|| {
                bad(format!(
                    "PDU {pdu_name} ({len} bytes at bit {offset}) does not fit in frame {name} \
                     ({size} bytes)"
                ))
            })?;
            self.pdu_signals(&instance.pdu_ref, pdu, &pdu_name, &place, &mut signals)?;
        }
        let mut count: HashMap<&str, Vec<&str>> = HashMap::new();
        for (pdu, signal, _) in &signals {
            count.entry(signal).or_default().push(pdu);
        }
        let clashing: Vec<String> = count
            .into_iter()
            .filter(|(_, pdus)| pdus.iter().any(|p| p != &pdus[0]))
            .map(|(s, _)| s.to_owned())
            .collect();
        let mut out: Vec<Signal> = Vec::with_capacity(signals.len());
        for (pdu, signal_name, mut signal) in signals {
            signal.name = if clashing.contains(&signal_name) {
                format!("{pdu}.{signal_name}")
            } else {
                signal_name
            };
            if out.iter().any(|s| s.name == signal.name) {
                return Err(bad(format!(
                    "frame {name} has the signal {} twice",
                    signal.name
                )));
            }
            out.push(signal);
        }
        Ok(Message {
            id: t.id,
            extended: t.extended,
            name,
            size,
            transmitters: self
                .ports
                .get(t.tid.as_str())
                .map(|p| p.outputs.clone())
                .unwrap_or_default(),
            receivers: self
                .ports
                .get(t.tid.as_str())
                .map(|p| p.inputs.clone())
                .unwrap_or_default(),
            signals: out,
            comment: frame.desc.clone(),
            cycle_time: t.cycle_time,
            fd: kind == BusKind::Can && (t.fd || size > 8),
        })
    }

    /// The signals of a PDU placed in a frame, as (PDU name, signal name,
    /// signal): its signal instances and, for a multiplexed PDU, the switch
    /// (the multiplexor), the signals of each switched PDU (multiplexed on its
    /// switch codes) and those of the static part (plain signals). Switched
    /// and static PDUs are laid out in the multiplexed PDU's bits, as
    /// AUTOSAR copies their segments.
    fn pdu_signals(
        &self,
        id: &str,
        pdu: &Pdu,
        pdu_name: &str,
        place: &Placement,
        out: &mut Vec<(String, String, Signal)>,
    ) -> Result<()> {
        let len = pdu.byte_length as usize;
        let add = |out: &mut Vec<_>,
                   owner: &str,
                   instances: &[SignalInstance],
                   mux: Option<Multiplexing>| {
            for s in instances {
                let (name, mut signal) = self.signal(owner, s, len)?;
                place.apply(&mut signal, len, owner, &name)?;
                signal.multiplexing = mux.clone();
                out.push((owner.to_owned(), name, signal));
            }
            Ok::<_, Error>(())
        };
        add(out, pdu_name, &pdu.signal_instances, None)?;
        let xml = self.ids.get(id).and_then(|n| child(*n, "MULTIPLEXER"));
        let Some(mux) = &pdu.multiplexer else {
            return match xml {
                Some(_) => Err(bad(format!(
                    "PDU {pdu_name}: its MULTIPLEXER has no DYNAMIC-PART with a SEGMENT-POSITION"
                ))),
                None => Ok(()),
            };
        };
        let what = format!("multiplexed PDU {pdu_name}");
        // The switch.
        let sw = &mux.switch;
        let switch_name = sw.short_name.clone().unwrap_or_else(|| "Switch".into());
        let order = if sw.is_high_low_byte_order {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        };
        if !bits::fits(sw.bit_position, sw.bit_length, order, len) {
            return Err(bad(format!(
                "{what}: the {} bits of switch {switch_name} from bit {} do not fit in {len} \
                 bytes",
                sw.bit_length, sw.bit_position
            )));
        }
        let mut switch = Signal {
            name: switch_name.clone(),
            start_bit: sw.bit_position,
            size: sw.bit_length,
            byte_order: order,
            signed: false,
            kind: ValueKind::Integer,
            factor: 1.0,
            offset: 0.0,
            minimum: None,
            maximum: None,
            unit: String::new(),
            receivers: Vec::new(),
            comment: None,
            choices: Vec::new(),
            initial_raw: None,
            is_multiplexor: true,
            multiplexing: None,
            coding: None,
        };
        place.apply(&mut switch, len, pdu_name, &switch_name)?;
        let multiplexor = out.len();
        out.push((pdu_name.to_owned(), switch_name, switch));
        // The dynamic part: switch codes by switched PDU, in code order.
        let switched = &mux.dynamic_part.pdu_instances_switch_map;
        let listed = xml
            .into_iter()
            .flat_map(|x| x.descendants())
            .filter(|n| n.is_element() && n.tag_name().name() == "SWITCHED-PDU-INSTANCE")
            .count();
        if listed != switched.len() {
            return Err(bad(format!(
                "{what}: every SWITCHED-PDU-INSTANCE needs a PDU-REF and a distinct integer \
                 SWITCH-CODE"
            )));
        }
        let mut codes: Vec<(&u64, &PduInstance)> = switched.iter().collect();
        codes.sort_by_key(|(code, _)| **code);
        let mut groups: Vec<(&str, Vec<(u64, u64)>)> = Vec::new();
        for (&code, instance) in codes {
            if sw.bit_length < 64 && code >> sw.bit_length != 0 {
                return Err(bad(format!(
                    "{what}: switch code {code} does not fit in the {} bits of switch {}",
                    sw.bit_length,
                    mux.switch.short_name.as_deref().unwrap_or("Switch")
                )));
            }
            self.check_inner_instance(
                &what,
                instance.bit_position,
                instance.is_high_low_byte_order,
            )?;
            let ranges = match groups.iter_mut().find(|(r, _)| *r == instance.pdu_ref) {
                Some((_, ranges)) => ranges,
                None => {
                    groups.push((&instance.pdu_ref, Vec::new()));
                    &mut groups.last_mut().expect("just pushed").1
                }
            };
            match ranges.last_mut() {
                Some((_, hi)) if *hi + 1 == code => *hi = code,
                _ => ranges.push((code, code)),
            }
        }
        for (pdu_ref, ranges) in groups {
            let (inner, inner_name) = self.inner_pdu(&what, pdu_ref)?;
            let mux = Multiplexing {
                multiplexor,
                ranges,
            };
            add(out, &inner_name, &inner.signal_instances, Some(mux))?;
        }
        // The static part.
        let statics = xml
            .into_iter()
            .flat_map(|x| children(x, "STATIC-PART"))
            .flat_map(|p| children(p, "STATIC-PDU-INSTANCE"));
        for instance in statics {
            let pdu_ref = child(instance, "PDU-REF")
                .and_then(|r| attr(r, "ID-REF"))
                .ok_or_else(|| bad(format!("{what}: a STATIC-PDU-INSTANCE has no PDU-REF")))?;
            let position = child(instance, "BIT-POSITION")
                .map(|p| number(p, &format!("{what} static part BIT-POSITION")))
                .transpose()?;
            let high_low = child(instance, "IS-HIGH-LOW-BYTE-ORDER").map(|h| text(h) == "true");
            self.check_inner_instance(&what, position.map(|p| p as u32), high_low)?;
            let (inner, inner_name) = self.inner_pdu(&what, pdu_ref)?;
            add(out, &inner_name, &inner.signal_instances, None)?;
        }
        Ok(())
    }

    /// A switched or static PDU of a multiplexed PDU, and its name.
    fn inner_pdu(&self, what: &str, id: &str) -> Result<(&Pdu, String)> {
        let pdu = self
            .data
            .elements
            .pdus_map_by_id
            .get(id)
            .ok_or_else(|| bad(format!("{what} refers to unknown PDU {id}")))?;
        let name = pdu.short_name.clone().unwrap_or_else(|| id.to_owned());
        if pdu.multiplexer.is_some() {
            return Err(bad(format!(
                "{what}: PDU {name} is multiplexed itself (nested multiplexing is not supported)"
            )));
        }
        Ok((pdu, name))
    }

    /// Switched and static PDU instances share the multiplexed PDU's bit
    /// numbering: a position other than 0 or a high-low order has no
    /// meaning there that this mapping could follow.
    fn check_inner_instance(
        &self,
        what: &str,
        position: Option<u32>,
        high_low: Option<bool>,
    ) -> Result<()> {
        if position.unwrap_or(0) != 0 || high_low == Some(true) {
            return Err(bad(format!(
                "{what}: switched and static PDU instances must be at bit 0, low-high (their \
                 signals use the multiplexed PDU's bit positions)"
            )));
        }
        Ok(())
    }

    /// A signal instance of a PDU of `len` bytes, positioned in the PDU.
    fn signal(
        &self,
        pdu: &str,
        instance: &afibex::fibex::SignalInstance,
        len: usize,
    ) -> Result<(String, Signal)> {
        let signal = self
            .data
            .elements
            .signals_map_by_id
            .get(&instance.signal_ref)
            .ok_or_else(|| {
                bad(format!(
                    "PDU {pdu} refers to unknown signal {}",
                    instance.signal_ref
                ))
            })?;
        let name = signal
            .short_name
            .clone()
            .unwrap_or_else(|| signal.id.clone());
        let what = format!("signal {name} of PDU {pdu}");
        let coded = self
            .data
            .pi
            .codings
            .get(&signal.coding_ref)
            .and_then(|c| c.coded_type.as_ref())
            .ok_or_else(|| {
                bad(format!(
                    "{what}: coding {} is missing or has no supported CODED-TYPE",
                    signal.coding_ref
                ))
            })?;
        if !matches!(coded.category, Category::StandardLengthType) {
            return Err(bad(format!(
                "{what}: only STANDARD-LENGTH-TYPE codings are supported"
            )));
        }
        let size = coded
            .bit_length
            .ok_or_else(|| bad(format!("{what}: the coding has no BIT-LENGTH")))?;
        let float = matches!(coded.encoding, Some(Encoding::IeeeFloating))
            || matches!(
                coded.base_data_type,
                Some(BaseDataType::AFloat32 | BaseDataType::AFloat64)
            );
        let (kind, signed) = if float {
            match size {
                32 => (ValueKind::Float32, true),
                64 => (ValueKind::Float64, true),
                _ => return Err(bad(format!("{what}: a float is 32 or 64 bits, not {size}"))),
            }
        } else {
            match &coded.encoding {
                Some(Encoding::Signed | Encoding::E2C) => (ValueKind::Integer, true),
                Some(Encoding::Unsigned | Encoding::Bit) => (ValueKind::Integer, false),
                Some(Encoding::SM) => (ValueKind::SignMagnitude, true),
                Some(Encoding::E1C) => (ValueKind::OnesComplement, true),
                None => (
                    ValueKind::Integer,
                    matches!(
                        coded.base_data_type,
                        Some(
                            BaseDataType::AInt8
                                | BaseDataType::AInt16
                                | BaseDataType::AInt32
                                | BaseDataType::AInt64
                        )
                    ),
                ),
                Some(e) => return Err(bad(format!("{what}: unsupported encoding {e:?}"))),
            }
        };
        let start_bit = instance
            .bit_position
            .ok_or_else(|| bad(format!("{what}: the signal instance has no BIT-POSITION")))?;
        let byte_order = if instance.is_high_low_byte_order.unwrap_or(false) {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        };
        if !bits::fits(start_bit, size, byte_order, len) {
            return Err(bad(format!(
                "{what}: {size} bits from bit {start_bit} do not fit in {len} bytes"
            )));
        }
        let compu = self.compu(&signal.coding_ref)?;
        let plain = compu.plain();
        let (factor, offset, choices) = match &plain {
            Some(p) => (p.factor, p.offset, p.choices.clone()),
            // The coding converts; the value table lists its names.
            None => (
                1.0,
                0.0,
                compu
                    .scales
                    .iter()
                    .filter_map(|s| match (&s.conversion, s.lower) {
                        (Conversion::Text(n), Bound::Included(l)) if l.fract() == 0.0 => {
                            Some((l as i64, n.clone()))
                        }
                        _ => None,
                    })
                    .collect(),
            ),
        };
        let mut signal = Signal {
            name: name.clone(),
            start_bit,
            size,
            byte_order,
            signed,
            kind,
            factor,
            offset,
            minimum: compu.minimum,
            maximum: compu.maximum,
            unit: compu.unit,
            receivers: Vec::new(),
            comment: signal.desc.clone(),
            choices,
            initial_raw: None,
            is_multiplexor: false,
            multiplexing: None,
            coding: None,
        };
        if plain.is_none() {
            let (lo, hi) = signal.raw_range();
            signal.coding = Some(Coding::new(compu.scales, lo, hi, kind.is_integer()));
        }
        Ok((name, signal))
    }

    /// The compu methods of a coding: IDENTICAL, LINEAR, SCALE-LINEAR,
    /// TEXTTABLE, RAT-FUNC and SCALE-RAT-FUNC.
    fn compu(&self, coding: &str) -> Result<Compu> {
        let mut c = Compu::default();
        let Some(node) = self.ids.get(coding).copied() else {
            return Err(bad(format!("coding {coding} not found")));
        };
        let methods = child(node, "COMPU-METHODS")
            .into_iter()
            .flat_map(|m| children(m, "COMPU-METHOD"));
        for method in methods {
            let category = child(method, "CATEGORY").map(text).unwrap_or_default();
            let unit = child(method, "UNIT-REF")
                .and_then(|u| attr(u, "ID-REF"))
                .and_then(|id| self.ids.get(id).copied());
            if let Some(unit) = unit {
                let name = child(unit, "DISPLAY-NAME").or_else(|| child(unit, "SHORT-NAME"));
                c.unit = name.map(text).unwrap_or_default().to_owned();
            }
            let constraint = child(method, "PHYS-CONSTRS")
                .into_iter()
                .flat_map(|p| children(p, "SCALE-CONSTR"))
                .find(|s| attr(*s, "VALIDITY").is_none_or(|v| v == "VALID"));
            if let Some(s) = constraint {
                c.minimum = limit(s, "LOWER-LIMIT", coding)?.or(c.minimum);
                c.maximum = limit(s, "UPPER-LIMIT", coding)?.or(c.maximum);
            }
            let scales: Vec<_> = child(method, "COMPU-INTERNAL-TO-PHYS")
                .and_then(|i| child(i, "COMPU-SCALES"))
                .into_iter()
                .flat_map(|s| children(s, "COMPU-SCALE"))
                .collect();
            let (names, functions, linear) = match category {
                "IDENTICAL" => {
                    c.scales.push(Scale {
                        lower: Bound::Unbounded,
                        upper: Bound::Unbounded,
                        conversion: Conversion::Rational {
                            numerator: vec![0.0, 1.0],
                            denominator: vec![1.0],
                        },
                    });
                    continue;
                }
                "LINEAR" => (false, true, true),
                "SCALE-LINEAR" => (true, true, true),
                "TEXTTABLE" => (true, false, false),
                "RAT-FUNC" => (false, true, false),
                "SCALE-RAT-FUNC" => (true, true, false),
                other => {
                    return Err(bad(format!(
                        "coding {coding}: unsupported compu method {other:?} (supported: \
                         IDENTICAL, LINEAR, SCALE-LINEAR, TEXTTABLE, RAT-FUNC, SCALE-RAT-FUNC)"
                    )))
                }
            };
            for scale in scales {
                let vt = child(scale, "COMPU-CONST").and_then(|k| child(k, "VT"));
                let coeffs = child(scale, "COMPU-RATIONAL-COEFFS");
                let conversion = match (vt, coeffs) {
                    (Some(vt), _) if names => Conversion::Text(text(vt).into()),
                    (None, Some(r)) if functions => rational(r, coding, linear)?,
                    _ => {
                        return Err(bad(format!(
                            "coding {coding}: unsupported {category} scale"
                        )))
                    }
                };
                c.scales.push(Scale {
                    lower: bound(scale, "LOWER-LIMIT", coding)?,
                    upper: bound(scale, "UPPER-LIMIT", coding)?,
                    conversion,
                });
            }
        }
        Ok(c)
    }
}

/// A scale constraint limit (`None` when absent or infinite).
fn limit(scale: Node<'_, '_>, name: &str, coding: &str) -> Result<Option<f64>> {
    match child(scale, name) {
        Some(l) if attr(l, "INTERVAL-TYPE") != Some("INFINITE") => {
            number(l, &format!("coding {coding} {name}")).map(Some)
        }
        _ => Ok(None),
    }
}

/// A compu scale's raw limit: closed unless `INTERVAL-TYPE` says `OPEN`,
/// unbounded when absent or `INFINITE`.
fn bound(scale: Node<'_, '_>, name: &str, coding: &str) -> Result<Bound<f64>> {
    let Some(l) = child(scale, name) else {
        return Ok(Bound::Unbounded);
    };
    let value = || number(l, &format!("coding {coding} {name}"));
    Ok(match attr(l, "INTERVAL-TYPE") {
        Some("INFINITE") => Bound::Unbounded,
        Some("OPEN") => Bound::Excluded(value()?),
        _ => Bound::Included(value()?),
    })
}

/// `COMPU-RATIONAL-COEFFS`: `N(x) / D(x)`, coefficients from the constant
/// term up. A `linear` method allows `(n0 + n1 x) / d0` only.
fn rational(coeffs: Node<'_, '_>, coding: &str, linear: bool) -> Result<Conversion> {
    let values = |name| -> Result<Vec<f64>> {
        child(coeffs, name)
            .into_iter()
            .flat_map(|n| children(n, "V"))
            .map(|v| number(v, &format!("coding {coding} {name}")))
            .collect()
    };
    let numerator = values("COMPU-NUMERATOR")?;
    let mut denominator = values("COMPU-DENOMINATOR")?;
    if denominator.is_empty() {
        denominator.push(1.0);
    }
    if numerator.is_empty() || denominator.iter().all(|d| *d == 0.0) {
        return Err(bad(format!(
            "coding {coding}: rational coefficients need a numerator and a non-zero denominator"
        )));
    }
    if linear && (numerator.len() > 2 || denominator.len() > 1 || denominator[0] == 0.0) {
        return Err(bad(format!(
            "coding {coding}: linear coefficients are at most two numerator values over one \
             non-zero denominator (a rational function is RAT-FUNC)"
        )));
    }
    Ok(Conversion::Rational {
        numerator,
        denominator,
    })
}

/// The CAN identifier and format of a frame triggering; `None` for one
/// without an identifier (not CAN).
fn triggering(t: Node<'_, '_>) -> Result<Option<Triggering>> {
    let tid = attr(t, "ID").unwrap_or_default();
    let Some(value) = t
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "IDENTIFIER-VALUE")
    else {
        return Ok(None);
    };
    let s = text(value);
    let id = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
    .filter(|&id| id <= 0x1FFF_FFFF)
    .ok_or_else(|| {
        bad(format!(
            "frame triggering {tid}: {s:?} is not a CAN identifier"
        ))
    })?;
    let has = |name: &str, value: &str| {
        t.descendants()
            .any(|n| n.is_element() && n.tag_name().name() == name && text(n) == value)
    };
    let extended = matches!(attr(value, "EXTENDED-ADDRESSING"), Some("true" | "1"))
        || has("CAN-ADDRESSING-MODE", "EXTENDED")
        || id > 0x7FF;
    let fd = has("CAN-FRAME-TX-BEHAVIOR", "CAN-FD") || has("CAN-FRAME-RX-BEHAVIOR", "CAN-FD");
    let frame = child(t, "FRAME-REF")
        .and_then(|f| attr(f, "ID-REF"))
        .ok_or_else(|| bad(format!("frame triggering {tid} has no FRAME-REF")))?;
    Ok(Some(Triggering {
        tid: tid.to_owned(),
        id,
        extended,
        fd,
        frame: frame.to_owned(),
        cycle_time: cycle_time(t, tid)?,
    }))
}

/// A number in decimal or `0x` hexadecimal.
fn integer(s: &str) -> Option<u64> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

/// A LIN frame triggering: the triggering, its frame identifier and
/// checksum model; `None` for one without an identifier. `classic` is the
/// cluster's default model (LIN 1.x).
fn lin_triggering(t: Node<'_, '_>, classic: bool) -> Result<Option<(Triggering, u8, LinChecksum)>> {
    let tid = attr(t, "ID").unwrap_or_default();
    let Some(value) = t
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "IDENTIFIER-VALUE")
    else {
        return Ok(None);
    };
    let s = text(value);
    let id = integer(s)
        .and_then(|i| u8::try_from(i).ok())
        .filter(|&i| i <= 0x3F)
        .ok_or_else(|| {
            bad(format!(
                "frame triggering {tid}: {s:?} is not a LIN frame identifier (0..=63)"
            ))
        })?;
    let declared = t
        .descendants()
        .find(|n| n.is_element() && matches!(n.tag_name().name(), "CHECKSUM-TYPE" | "LIN-CHECKSUM"))
        .map(|n| match text(n).to_ascii_uppercase().as_str() {
            "CLASSIC" => Ok(LinChecksum::Classic),
            "ENHANCED" => Ok(LinChecksum::Enhanced),
            other => Err(bad(format!(
                "frame triggering {tid}: checksum type {other:?} is not CLASSIC or ENHANCED"
            ))),
        })
        .transpose()?;
    let model = if crate::message::is_lin_diagnostic(id) {
        LinChecksum::Classic
    } else {
        declared.unwrap_or(if classic {
            LinChecksum::Classic
        } else {
            LinChecksum::Enhanced
        })
    };
    let frame = child(t, "FRAME-REF")
        .and_then(|f| attr(f, "ID-REF"))
        .ok_or_else(|| bad(format!("frame triggering {tid} has no FRAME-REF")))?;
    Ok(Some((
        Triggering {
            tid: tid.to_owned(),
            id: u32::from(id),
            extended: false,
            fd: false,
            frame: frame.to_owned(),
            cycle_time: cycle_time(t, tid)?,
        },
        id,
        model,
    )))
}

/// The `SCHEDULE-TABLE`s of a LIN channel and its cluster, with their IDs.
/// An entry is an element of the table with a `DELAY` child; its
/// `FRAME-TRIGGERING-REF` must be one of the channel's (`frames`: frame
/// names by triggering ID).
fn schedule_tables(
    channel: Node<'_, '_>,
    cluster: Option<Node<'_, '_>>,
    frames: &HashMap<String, String>,
) -> Result<Vec<(String, ScheduleTable)>> {
    let mut out = Vec::new();
    let nodes = channel
        .descendants()
        .chain(cluster.into_iter().flat_map(|c| c.descendants()))
        .filter(|n| n.is_element() && n.tag_name().name() == "SCHEDULE-TABLE");
    for table in nodes {
        let id = attr(table, "ID").unwrap_or_default().to_owned();
        let name = child(table, "SHORT-NAME")
            .map(text)
            .filter(|n| !n.is_empty())
            .unwrap_or(&id)
            .to_owned();
        let mut entries = Vec::new();
        for entry in table.descendants().filter(|n| n.is_element()) {
            let Some(delay) = child(entry, "DELAY") else {
                continue;
            };
            let s = text(delay);
            let delay = xs_duration(s).ok_or_else(|| {
                bad(format!(
                    "schedule table {name}: delay {s:?} is not a duration such as PT0.01S"
                ))
            })?;
            let frame = match child(entry, "FRAME-TRIGGERING-REF").and_then(|r| attr(r, "ID-REF")) {
                None => None,
                Some(r) => Some(frames.get(r).cloned().ok_or_else(|| {
                    bad(format!(
                        "schedule table {name} refers to frame triggering {r}, which is not \
                         a LIN frame triggering of the channel"
                    ))
                })?),
            };
            entries.push(ScheduleEntry { frame, delay });
        }
        out.push((id, ScheduleTable { name, entries }));
    }
    Ok(out)
}

/// The cycle time of a frame triggering: the `VALUE` of the
/// `REPEATING-TIME-RANGE` of its first `TIMINGS/CYCLIC-TIMING`; `None`
/// without one, or for a zero time (as a DBC `GenMsgCycleTime` of 0).
fn cycle_time(t: Node<'_, '_>, tid: &str) -> Result<Option<Duration>> {
    let Some(value) = children(t, "TIMINGS")
        .flat_map(|n| children(n, "CYCLIC-TIMING"))
        .find_map(|n| child(n, "REPEATING-TIME-RANGE"))
        .and_then(|n| child(n, "VALUE"))
    else {
        return Ok(None);
    };
    let s = text(value);
    let d = xs_duration(s).ok_or_else(|| {
        bad(format!(
            "frame triggering {tid}: cycle time {s:?} is not a duration of days, hours, \
             minutes and seconds such as PT0.01S"
        ))
    })?;
    Ok((!d.is_zero()).then_some(d))
}

/// An XML Schema `xs:duration` of days, hours, minutes and (fractional)
/// seconds, such as `PT0.01S` or `P1DT2H`. Years and months, which have no
/// fixed length, and negative durations are `None`.
fn xs_duration(s: &str) -> Option<Duration> {
    let rest = s.strip_prefix('P')?;
    let (date, time) = match rest.split_once('T') {
        Some((d, t)) if !t.is_empty() => (d, Some(t)),
        Some(_) => return None,
        None => (rest, None),
    };
    if date.is_empty() && time.is_none() {
        return None;
    }
    let mut total = Duration::ZERO;
    // Components in order, each at most once: (designator, seconds each,
    // whether a fraction is allowed).
    let mut parse = |part: &str, units: &[(char, u64, bool)]| -> Option<()> {
        let mut part = part;
        let mut units = units.iter();
        while !part.is_empty() {
            let end = part.find(|c: char| !c.is_ascii_digit() && c != '.')?;
            let (number, designator) = (&part[..end], part[end..].chars().next()?);
            let &(_, seconds, fraction) = units.find(|(d, _, _)| *d == designator)?;
            if number.is_empty() || (!fraction && number.contains('.')) {
                return None;
            }
            let (whole, frac) = number.split_once('.').unwrap_or((number, ""));
            if whole.is_empty() || frac.contains('.') {
                return None;
            }
            // Exact decimal: whole seconds, then the fraction in nanoseconds.
            let whole: u64 = whole.parse().ok()?;
            let nanos: u64 = format!("{:0<9}", frac.get(..frac.len().min(9))?)
                .parse()
                .ok()?;
            let part_total = Duration::from_secs(whole.checked_mul(seconds)?)
                .checked_add(Duration::from_nanos(nanos.checked_mul(seconds)?))?;
            total = total.checked_add(part_total)?;
            part = &part[end + 1..];
        }
        Some(())
    };
    parse(date, &[('D', 86_400, false)])?;
    if let Some(time) = time {
        parse(
            time,
            &[('H', 3_600, false), ('M', 60, false), ('S', 1, true)],
        )?;
    }
    Some(total)
}

#[cfg(test)]
mod tests;
