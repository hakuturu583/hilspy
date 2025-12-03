import asyncio
import logging
import struct
import time
import tempfile
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import v4l2  # type: ignore
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import datetime

import av
import cv2
import numpy as np
import v4l2  # type: ignore
from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.asyncio.client import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived, QuicEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class V4L2Camera:
    def __init__(
        self,
        device_path: str = "/dev/video0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        bitrate: int = 2000000,
    ):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.v4l2_output = None
        self.encoder = None
        self.decoder = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False

    def _setup_h264_encoder(self) -> av.CodecContext:
        codec = av.codec.Codec("libx264", "w")
        encoder = codec.create()
        encoder.width = self.width  # type: ignore
        encoder.height = self.height  # type: ignore
        encoder.pix_fmt = "yuv420p"  # type: ignore
        encoder.bit_rate = self.bitrate
        encoder.framerate = self.fps  # type: ignore
        encoder.gop_size = self.fps  # type: ignore
        from fractions import Fraction

        encoder.time_base = Fraction(1, self.fps)
        encoder.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
        }
        encoder.open()
        return encoder

    def _setup_h264_decoder(self) -> av.CodecContext:
        codec = av.codec.Codec("h264", "r")
        decoder = codec.create()
        decoder.open()
        return decoder

    def _setup_v4l2_output(self):
        import fcntl

        self.v4l2_fd = open(self.device_path, "wb")

        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_BGR24
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fmt.fmt.pix.bytesperline = self.width * 3
        fmt.fmt.pix.sizeimage = self.width * self.height * 3
        fmt.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_SRGB

        fcntl.ioctl(self.v4l2_fd, v4l2.VIDIOC_S_FMT, fmt)

    def _setup_camera_capture(self, input_device: str = "/dev/video1"):
        self.capture = cv2.VideoCapture(input_device)
        if self.capture is not None and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

    def encode_frame(self, frame: np.ndarray) -> bytes:
        if self.encoder is None:
            return b""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts = int(time.time() * self.fps)
        from fractions import Fraction

        av_frame.time_base = Fraction(1, self.fps)

        packets = self.encoder.encode(av_frame)
        encoded_data = b""
        for packet in packets:
            encoded_data += bytes(packet)

        return encoded_data

    def decode_frame(self, data: bytes) -> Optional[np.ndarray]:
        if self.decoder is None:
            return None
        try:
            packet = av.Packet(data)
            frames = self.decoder.decode(packet)

            for frame in frames:
                img = frame.to_ndarray(format="bgr24")
                return img
            return None
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None

    def write_to_v4l2(self, frame: np.ndarray):
        if frame.shape != (self.height, self.width, 3):
            frame = cv2.resize(frame, (self.width, self.height))

        self.v4l2_fd.write(frame.tobytes())
        self.v4l2_fd.flush()


class QuicCameraServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, camera: V4L2Camera, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = camera
        self.stream_id = None

    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            if event.end_stream:
                return

            frame = self.camera.decode_frame(event.data)
            if frame is not None:
                self.camera.write_to_v4l2(frame)
                logger.debug("Received and wrote frame to v4l2")


class QuicCameraClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, camera: V4L2Camera, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = camera
        self.stream_id = None
        self.connected = False

    def quic_event_received(self, event: QuicEvent):
        pass


def create_self_signed_cert():
    """Create a self-signed certificate in memory"""
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.DNSName("127.0.0.1"),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return cert_pem, key_pem


class V4L2CameraServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4433,
        camera: Optional[V4L2Camera] = None,
    ):
        self.host = host
        self.port = port
        self.camera = camera or V4L2Camera()
        self._cert_file = None
        self._key_file = None

    async def run(self):
        self.camera.encoder = self.camera._setup_h264_encoder()
        self.camera.decoder = self.camera._setup_h264_decoder()
        self.camera._setup_v4l2_output()

        configuration = QuicConfiguration(
            alpn_protocols=["h264-stream"],
            is_client=False,
        )

        # Create temporary self-signed certificate
        cert_pem, key_pem = create_self_signed_cert()
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pem"
        ) as cert_file:
            cert_file.write(cert_pem)
            self._cert_file = cert_file.name

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pem"
        ) as key_file:
            key_file.write(key_pem)
            self._key_file = key_file.name

        configuration.load_cert_chain(self._cert_file, self._key_file)

        logger.info(f"Starting QUIC camera server on {self.host}:{self.port}")

        try:
            await serve(
                self.host,
                self.port,
                configuration=configuration,
                create_protocol=lambda *args, **kwargs: QuicCameraServerProtocol(
                    *args, camera=self.camera, **kwargs
                ),
            )

            await asyncio.Future()
        finally:
            # Clean up temp files
            import os

            if self._cert_file and os.path.exists(self._cert_file):
                os.unlink(self._cert_file)
            if self._key_file and os.path.exists(self._key_file):
                os.unlink(self._key_file)


class V4L2CameraClient:
    """
    QUIC client for sending video frames to V4L2CameraServer.

    This client accepts image data via Python API instead of using a camera device.

    Example usage:
        client = V4L2CameraClient("localhost", 4433)
        await client.connect()

        # Send frames via Python API
        for frame in your_frame_source:
            await client.send_frame(frame)
    """

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 4433,
        camera: Optional[V4L2Camera] = None,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.camera = camera or V4L2Camera()
        self._frame_queue: asyncio.Queue = asyncio.Queue()
        self._protocol: Optional[QuicCameraClientProtocol] = None
        self._stream_id: Optional[int] = None
        self._running = False

    async def connect(self):
        """Connect to the QUIC server"""
        self.camera.encoder = self.camera._setup_h264_encoder()

        configuration = QuicConfiguration(
            alpn_protocols=["h264-stream"],
            is_client=True,
        )
        configuration.verify_mode = False

        self._connection = await connect(
            self.server_host,
            self.server_port,
            configuration=configuration,
            create_protocol=lambda *args, **kwargs: QuicCameraClientProtocol(
                *args, camera=self.camera, **kwargs
            ),
        )
        self._protocol = self._connection
        self._stream_id = self._protocol._quic.get_next_available_stream_id()
        self._running = True

        logger.info(
            f"Connected to QUIC server at {self.server_host}:{self.server_port}"
        )

    async def send_frame(self, frame: np.ndarray):
        """Send a frame to the server via Python API"""
        if not self._running or self._protocol is None or self._stream_id is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        await self._frame_queue.put(frame)

    async def _frame_sender(self):
        """Background task to send frames from the queue"""
        while self._running:
            try:
                frame = await asyncio.wait_for(self._frame_queue.get(), timeout=0.1)

                encoded_data = self.camera.encode_frame(frame)
                if encoded_data and self._protocol and self._stream_id is not None:
                    size_header = struct.pack("!I", len(encoded_data))
                    self._protocol._quic.send_stream_data(
                        self._stream_id, size_header + encoded_data, end_stream=False
                    )
                    self._protocol.transmit()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error sending frame: {e}")

    async def run(self):
        """Start the client and begin sending frames"""
        await self.connect()

        # Start the background frame sender
        sender_task = asyncio.create_task(self._frame_sender())

        try:
            await sender_task
        except asyncio.CancelledError:
            self._running = False
            sender_task.cancel()
            await self._connection.close()

    def stop(self):
        """Stop the client"""
        self._running = False


async def main_server():
    camera = V4L2Camera(
        device_path="/dev/video0", width=640, height=480, fps=30, bitrate=2000000
    )

    server = V4L2CameraServer(host="0.0.0.0", port=4433, camera=camera)

    await server.run()


async def main_client():
    """Example usage of V4L2CameraClient with Python API"""
    camera = V4L2Camera(width=640, height=480, fps=30, bitrate=2000000)

    client = V4L2CameraClient(
        server_host="localhost",
        server_port=4433,
        camera=camera,
    )

    # Connect to server
    await client.connect()

    # Start the frame sender task
    sender_task = asyncio.create_task(client.run())

    # Example: Send frames from camera device (for demonstration)
    # In real usage, you would call client.send_frame() with your image data
    cap = cv2.VideoCapture("/dev/video1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # Send frame via Python API
                await client.send_frame(frame)
            await asyncio.sleep(1.0 / 30)  # 30 FPS
    except KeyboardInterrupt:
        client.stop()
        sender_task.cancel()
        cap.release()
        print("Client stopped")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        asyncio.run(main_server())
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        asyncio.run(main_client())
    else:
        print("Usage: python v4l2_camera.py [server|client]")
        sys.exit(1)
