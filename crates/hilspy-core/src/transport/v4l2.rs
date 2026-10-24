//! `v4l2://DEVICE`: video frames into a V4L2 output device (Linux), such as
//! a `v4l2loopback` camera, through GStreamer:
//!
//! ```text
//! appsrc ! [jpegdec | pngdec] ! videoconvert ! videoscale ! video/x-raw,format=OUTPUT ! v4l2sink
//! ```
//!
//! Every sent packet is one frame. `v4l2:///dev/video10?width=640&height=480`.
//! Options:
//!
//! * `format`: what a packet holds, a GStreamer raw video format (`RGB`, the
//!   default, `BGR`, `RGBA`, `BGRA`, `GRAY8`, `I420`, `NV12`, `YUY2`, ...) or
//!   an encoded image, `jpeg` or `png`;
//! * `width`, `height`: the frame size, required for raw formats; for
//!   `jpeg` / `png` the size to scale to (default: the image's);
//! * `fps`: the nominal frame rate written to the caps (`30`, the default,
//!   or `30000/1001`); frames go out as they are sent, the caller paces them;
//! * `output`: the raw format the device receives (default `YUY2`, which
//!   browsers, OpenCV and ROS camera drivers read).
//!
//! A raw frame is either tightly packed (`width * height * bytes per pixel`,
//! as `numpy.ndarray.tobytes()` gives) or laid out with GStreamer's default
//! strides. It receives nothing. A pipeline error (the device refuses the
//! caps, a JPEG that does not decode) fails the send that reports it.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use super::Transport;
use crate::error::{Error, Result};
use crate::message::Packet;
use crate::url::Endpoint;
use crate::value::Options;

/// What a packet holds.
enum Input {
    Raw(gst_video::VideoInfo),
    Jpeg,
    Png,
}

pub(crate) struct V4l2 {
    pipeline: gst::Pipeline,
    src: gst_app::AppSrc,
    input: Input,
    /// The first pipeline error, reported by every later send.
    failed: Mutex<Option<String>>,
    closed: Mutex<bool>,
    wake: Condvar,
}

fn take_u32(o: &mut Options, key: &str) -> Result<Option<u32>> {
    o.take_str(key)?
        .map(|v| {
            v.parse::<u32>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| Error::Config(format!("v4l2 option {key}: {v:?} is not a size")))
        })
        .transpose()
}

fn fraction(v: &str) -> Option<gst::Fraction> {
    let (n, d) = v.split_once('/').unwrap_or((v, "1"));
    let (n, d) = (n.parse::<i32>().ok()?, d.parse::<i32>().ok()?);
    (n > 0 && d > 0).then(|| gst::Fraction::new(n, d))
}

fn raw_format(key: &str, name: &str) -> Result<gst_video::VideoFormat> {
    match gst_video::VideoFormat::from_string(name) {
        gst_video::VideoFormat::Unknown | gst_video::VideoFormat::Encoded => Err(Error::Config(
            format!("v4l2 option {key}: {name:?} is not a GStreamer raw video format (RGB, GRAY8, YUY2, ...)"),
        )),
        f => Ok(f),
    }
}

fn element(factory: &str) -> Result<gst::Element> {
    gst::ElementFactory::make(factory).build().map_err(|_| {
        Error::Config(format!(
            "GStreamer element {factory} is missing: install the GStreamer base and good \
             plugins (tools/setup_host.py does)"
        ))
    })
}

fn gst_error(e: impl std::fmt::Display) -> Error {
    Error::Config(format!("GStreamer: {e}"))
}

impl V4l2 {
    pub(crate) fn open(endpoint: Endpoint) -> Result<Self> {
        if endpoint.address.is_empty() {
            return Err(Error::Config(
                "v4l2:// needs a device: v4l2:///dev/video10".into(),
            ));
        }
        gst::init().map_err(gst_error)?;
        let sink = element("v4l2sink")?;
        sink.set_property("device", &endpoint.address);
        Self::with_sink(endpoint.options, sink)
    }

    /// The pipeline into `sink` (`v4l2sink`, or an `appsink` in tests).
    fn with_sink(mut o: Options, sink: gst::Element) -> Result<Self> {
        gst::init().map_err(gst_error)?;
        let format = o.take_str("format")?.unwrap_or_else(|| "RGB".into());
        let width = take_u32(&mut o, "width")?;
        let height = take_u32(&mut o, "height")?;
        let fps = match o.take_str("fps")? {
            None => gst::Fraction::new(30, 1),
            Some(v) => fraction(&v).ok_or_else(|| {
                Error::Config(format!(
                    "v4l2 option fps: {v:?} is not a rate (30, 30000/1001)"
                ))
            })?,
        };
        let output = raw_format(
            "output",
            &o.take_str("output")?.unwrap_or_else(|| "YUY2".into()),
        )?;
        o.finish("v4l2://")?;

        let (input, caps) = match format.to_ascii_lowercase().as_str() {
            "jpeg" | "jpg" => (
                Input::Jpeg,
                gst::Caps::builder("image/jpeg")
                    .field("framerate", fps)
                    .build(),
            ),
            "png" => (
                Input::Png,
                gst::Caps::builder("image/png")
                    .field("framerate", fps)
                    .build(),
            ),
            _ => {
                let (Some(w), Some(h)) = (width, height) else {
                    return Err(Error::Config(format!(
                        "v4l2:// format {format} needs width and height"
                    )));
                };
                let info = gst_video::VideoInfo::builder(raw_format("format", &format)?, w, h)
                    .fps(fps)
                    .build()
                    .map_err(gst_error)?;
                let caps = info.to_caps().map_err(gst_error)?;
                (Input::Raw(info), caps)
            }
        };
        let mut out = gst::Caps::builder("video/x-raw")
            .field("format", output.to_str())
            .field("framerate", fps);
        if let Some(w) = width {
            out = out.field("width", w as i32);
        }
        if let Some(h) = height {
            out = out.field("height", h as i32);
        }

        let src = gst_app::AppSrc::builder()
            .caps(&caps)
            .format(gst::Format::Time)
            .is_live(true)
            .do_timestamp(true)
            // Back-pressure instead of an unbounded queue: a send waits
            // while two frames are in flight.
            .block(true)
            .max_bytes(match &input {
                Input::Raw(info) => 2 * info.size() as u64,
                _ => 0,
            })
            .build();
        let filter = element("capsfilter")?;
        filter.set_property("caps", out.build());
        if sink.has_property("sync") {
            sink.set_property("sync", false);
        }
        let mut chain = vec![src.clone().upcast::<gst::Element>()];
        match input {
            Input::Jpeg => chain.push(element("jpegdec")?),
            Input::Png => chain.push(element("pngdec")?),
            Input::Raw(_) => {}
        }
        chain.extend([
            element("videoconvert")?,
            element("videoscale")?,
            filter,
            sink,
        ]);

        let pipeline = gst::Pipeline::new();
        pipeline.add_many(&chain).map_err(gst_error)?;
        gst::Element::link_many(&chain).map_err(gst_error)?;
        let this = Self {
            pipeline,
            src,
            input,
            failed: Mutex::new(None),
            closed: Mutex::new(false),
            wake: Condvar::new(),
        };
        if this.pipeline.set_state(gst::State::Playing).is_err() {
            // Opening the device fails here; the bus says why.
            let why = this
                .bus_error()
                .unwrap_or_else(|| "the pipeline does not start".into());
            return Err(Error::Config(why));
        }
        Ok(this)
    }

    /// The first error the pipeline posted, if any.
    fn bus_error(&self) -> Option<String> {
        let msg = self
            .pipeline
            .bus()?
            .pop_filtered(&[gst::MessageType::Error])?;
        match msg.view() {
            gst::MessageView::Error(e) => Some(match e.debug() {
                Some(d) => format!("{} ({d})", e.error()),
                None => e.error().to_string(),
            }),
            _ => None,
        }
    }

    fn check(&self) -> Result<()> {
        let mut failed = self.failed.lock().unwrap();
        if failed.is_none() {
            *failed = self.bus_error();
        }
        match &*failed {
            Some(e) => Err(Error::Io(std::io::Error::other(format!("v4l2: {e}")))),
            None => Ok(()),
        }
    }

    /// The packet as a buffer of the input caps.
    fn buffer(&self, data: &[u8]) -> Result<gst::Buffer> {
        let Input::Raw(info) = &self.input else {
            return Ok(gst::Buffer::from_slice(data.to_vec()));
        };
        if data.len() == info.size() {
            return Ok(gst::Buffer::from_slice(data.to_vec()));
        }
        // Tightly packed rows of a single-plane format, padded to the strides.
        let finfo = info.format_info();
        let row = info.width() as usize * finfo.pixel_stride()[0] as usize;
        let stride = info.stride()[0] as usize;
        let rows = info.height() as usize;
        if info.n_planes() != 1 || data.len() != row * rows {
            return Err(Error::InvalidPacket(format!(
                "a {}x{} {} frame is {} bytes, not {}",
                info.width(),
                info.height(),
                info.format().to_str(),
                if info.n_planes() == 1 && row != stride {
                    format!("{} (or {})", row * rows, info.size())
                } else {
                    info.size().to_string()
                },
                data.len()
            )));
        }
        let mut padded = vec![0u8; info.size()];
        for (dst, src) in padded.chunks_mut(stride).zip(data.chunks(row)) {
            dst[..row].copy_from_slice(src);
        }
        Ok(gst::Buffer::from_slice(padded))
    }
}

impl Transport for V4l2 {
    fn send(&self, packet: &Packet) -> Result<()> {
        if *self.closed.lock().unwrap() {
            return Err(Error::Closed);
        }
        self.check()?;
        let buffer = self.buffer(&packet.data)?;
        match self.src.push_buffer(buffer) {
            Ok(_) => self.check(),
            Err(gst::FlowError::Flushing) | Err(gst::FlowError::Eos)
                if *self.closed.lock().unwrap() =>
            {
                Err(Error::Closed)
            }
            Err(e) => {
                self.check()?;
                Err(Error::Io(std::io::Error::other(format!("v4l2: {e:?}"))))
            }
        }
    }

    /// A V4L2 output receives nothing: wait out the timeout.
    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let closed = self.closed.lock().unwrap();
        let closed = match timeout {
            None => self.wake.wait_while(closed, |c| !*c).unwrap(),
            Some(t) => self.wake.wait_timeout_while(closed, t, |c| !*c).unwrap().0,
        };
        if *closed {
            Err(Error::Closed)
        } else {
            Ok(None)
        }
    }

    fn close(&self) {
        let mut closed = self.closed.lock().unwrap();
        if !*closed {
            *closed = true;
            let _ = self.pipeline.set_state(gst::State::Null);
        }
        self.wake.notify_all();
    }
}

impl Drop for V4l2 {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(query: &str) -> Options {
        Endpoint::parse(&format!("v4l2:///dev/null?{query}"))
            .unwrap()
            .options
    }

    /// A transport into an appsink, and the sink to pull frames from.
    fn open(query: &str) -> (V4l2, gst_app::AppSink) {
        gst::init().unwrap();
        let sink = gst_app::AppSink::builder().sync(false).build();
        let t = V4l2::with_sink(options(query), sink.clone().upcast()).unwrap();
        (t, sink)
    }

    fn pull(sink: &gst_app::AppSink) -> (gst::Caps, Vec<u8>) {
        let sample = sink
            .try_pull_sample(gst::ClockTime::from_seconds(5))
            .expect("a frame");
        let data = sample.buffer().unwrap().map_readable().unwrap().to_vec();
        (sample.caps().unwrap().to_owned(), data)
    }

    fn size(caps: &gst::Caps) -> (i32, i32, String) {
        let s = caps.structure(0).unwrap();
        (
            s.get("width").unwrap(),
            s.get("height").unwrap(),
            s.get("format").unwrap(),
        )
    }

    #[test]
    fn gray_frames_pass_unchanged() {
        let (t, sink) = open("format=GRAY8&output=GRAY8&width=8&height=4");
        let frame: Vec<u8> = (0..32).collect();
        t.send(&Packet::bytes(frame.clone())).unwrap();
        let (caps, data) = pull(&sink);
        assert_eq!(size(&caps), (8, 4, "GRAY8".into()));
        assert_eq!(data, frame);
        t.close();
        assert!(matches!(t.send(&Packet::bytes(frame)), Err(Error::Closed)));
    }

    #[test]
    fn rgb_becomes_yuy2() {
        // 5 pixels wide: GStreamer pads RGB rows to 16 bytes, numpy does not.
        let (t, sink) = open("width=5&height=2&fps=30000/1001");
        t.send(&Packet::bytes(vec![255u8; 5 * 2 * 3])).unwrap();
        let (caps, data) = pull(&sink);
        assert_eq!(size(&caps), (5, 2, "YUY2".into()));
        let rate: gst::Fraction = caps.structure(0).unwrap().get("framerate").unwrap();
        assert_eq!(rate, gst::Fraction::new(30000, 1001));
        // White: luma at the top of the video range, chroma neutral.
        let row = &data[..8];
        assert!(row.chunks(2).all(|p| p[0] >= 230), "{row:?}");
        assert!(row
            .chunks(4)
            .all(|p| p[1].abs_diff(128) <= 2 && p[3].abs_diff(128) <= 2));
        // The padded layout is accepted as well; anything else is not.
        t.send(&Packet::bytes(vec![0u8; 16 * 2])).unwrap();
        pull(&sink);
        let err = t
            .send(&Packet::bytes(vec![0u8; 7]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("30 (or 32)"), "{err}");
    }

    #[test]
    fn jpeg_is_decoded_and_scaled() {
        gst::init().unwrap();
        let jpeg = gst::parse::launch(
            "videotestsrc num-buffers=1 ! video/x-raw,width=64,height=48 ! jpegenc ! appsink name=out",
        )
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let out = jpeg
            .by_name("out")
            .unwrap()
            .downcast::<gst_app::AppSink>()
            .unwrap();
        jpeg.set_state(gst::State::Playing).unwrap();
        let (_, image) = pull(&out);
        jpeg.set_state(gst::State::Null).unwrap();
        assert_eq!(&image[..2], b"\xff\xd8");

        let (t, sink) = open("format=jpeg");
        t.send(&Packet::bytes(image.clone())).unwrap();
        let (caps, data) = pull(&sink);
        assert_eq!(size(&caps), (64, 48, "YUY2".into()));
        assert_eq!(data.len(), 64 * 48 * 2);

        let (t, sink) = open("format=jpeg&width=32&height=24&output=RGB");
        t.send(&Packet::bytes(image)).unwrap();
        assert_eq!(size(&pull(&sink).0), (32, 24, "RGB".into()));
    }

    #[test]
    fn a_broken_image_fails_a_send() {
        let (t, _sink) = open("format=png");
        let mut result = Ok(());
        for _ in 0..50 {
            result = t.send(&Packet::bytes(b"not a png".to_vec()));
            if result.is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(result.is_err());
        // It stays failed.
        assert!(t.send(&Packet::bytes(b"x".to_vec())).is_err());
    }

    #[test]
    fn receives_nothing() {
        let (t, _sink) = open("format=GRAY8&width=2&height=2");
        assert!(t.recv(Some(Duration::from_millis(20))).unwrap().is_none());
        std::thread::scope(|s| {
            let r = s.spawn(|| t.recv(None));
            std::thread::sleep(Duration::from_millis(20));
            t.close();
            assert!(matches!(r.join().unwrap(), Err(Error::Closed)));
        });
    }

    #[test]
    fn bad_options() {
        let err = |url: &str| {
            V4l2::open(Endpoint::parse(url).unwrap())
                .err()
                .unwrap()
                .to_string()
        };
        assert!(err("v4l2://?width=2&height=2").contains("needs a device"));
        assert!(err("v4l2:///dev/video99?format=RGB").contains("needs width and height"));
        assert!(err("v4l2:///dev/video99?format=XYZ&width=2&height=2").contains("format"));
        assert!(err("v4l2:///dev/video99?output=jpeg").contains("output"));
        assert!(err("v4l2:///dev/video99?width=0&height=2").contains("width"));
        assert!(err("v4l2:///dev/video99?fps=fast&width=2&height=2").contains("fps"));
        assert!(err("v4l2:///dev/video99?typo&width=2&height=2").contains("typo"));
        // A device that is not there fails at open.
        let e = err("v4l2:///nonexistent/video0?width=2&height=2");
        assert!(e.contains("nonexistent"), "{e}");
    }
}
