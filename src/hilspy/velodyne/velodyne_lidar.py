import asyncio
import socket
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, TypeVar, Generic
import logging
from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration

logger = logging.getLogger(__name__)


@dataclass
class LaserReturn:
    distance: int  # Distance in 2mm units
    intensity: int  # Intensity 0-255


@dataclass
class DataBlock:
    flag: int  # 0xEEFF for upper block, 0xDDFF for lower block
    azimuth: float  # Azimuth angle in degrees
    laser_returns: List[LaserReturn]  # 32 laser returns (2 firings * 16 lasers)


@dataclass
class VelodynePacket:
    PACKET_SIZE = 1206
    BLOCKS_PER_PACKET = 12
    LASERS_PER_BLOCK = 32
    BLOCK_SIZE = 100  # 2 (flag) + 2 (azimuth) + 32*3 (laser data)

    blocks: List[DataBlock]
    timestamp: int  # Microseconds since top of hour
    factory: bytes  # 2 bytes for return mode and product ID

    @classmethod
    def from_bytes(cls, data: bytes) -> "VelodynePacket":
        if len(data) != cls.PACKET_SIZE:
            raise ValueError(
                f"Invalid packet size: {len(data)} (expected {cls.PACKET_SIZE})"
            )

        blocks = []
        offset = 0

        for _ in range(cls.BLOCKS_PER_PACKET):
            flag = struct.unpack("<H", data[offset : offset + 2])[0]
            offset += 2

            azimuth = struct.unpack("<H", data[offset : offset + 2])[0] / 100.0
            offset += 2

            laser_returns = []
            for _ in range(cls.LASERS_PER_BLOCK):
                distance = struct.unpack("<H", data[offset : offset + 2])[0]
                offset += 2
                intensity = data[offset]
                offset += 1
                laser_returns.append(LaserReturn(distance, intensity))

            blocks.append(DataBlock(flag, azimuth, laser_returns))

        timestamp = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4

        factory = data[offset : offset + 2]

        return cls(blocks=blocks, timestamp=timestamp, factory=factory)

    def to_bytes(self) -> bytes:
        data = bytearray()

        for block in self.blocks:
            data += struct.pack("<H", block.flag)
            data += struct.pack("<H", int(block.azimuth * 100))

            for laser in block.laser_returns:
                data += struct.pack("<H", laser.distance)
                data += struct.pack("B", laser.intensity)

        data += struct.pack("<I", self.timestamp)
        data += self.factory

        # Pad to exact packet size if needed
        while len(data) < self.PACKET_SIZE:
            data.append(0)

        return bytes(data)

    @classmethod
    def create_empty(cls) -> "VelodynePacket":
        blocks = []
        for i in range(cls.BLOCKS_PER_PACKET):
            flag = 0xEEFF if i % 2 == 0 else 0xDDFF
            laser_returns = [LaserReturn(0, 0) for _ in range(cls.LASERS_PER_BLOCK)]
            blocks.append(DataBlock(flag, 0.0, laser_returns))

        return cls(blocks=blocks, timestamp=0, factory=b"\x00\x00")


# Type variable for packet types
PacketType = TypeVar("PacketType", bound=VelodynePacket)


class VelodyneLidar(ABC, Generic[PacketType]):
    def __init__(
        self,
        packet_class: type[PacketType],
        quic_host: str = "localhost",
        quic_port: int = 4433,
        udp_host: str = "127.0.0.1",
        udp_port: int = 2368,
        stream_id: int = 0,
    ):
        self.quic_host = quic_host
        self.quic_port = quic_port
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.stream_id = stream_id
        self.packet_class = packet_class

        self.udp_socket: Optional[socket.socket] = None
        self.quic_protocol: Optional[QuicConnectionProtocol] = None
        self.packet_queue: asyncio.Queue[PacketType] = asyncio.Queue()
        self.running = False

    async def connect(self):
        logger.info(f"Connecting to QUIC server at {self.quic_host}:{self.quic_port}")

        configuration = QuicConfiguration(
            is_client=True,
            alpn_protocols=["h3"],
            verify_mode=False,  # Set to True in production with proper certificates
        )

        async with connect(
            self.quic_host,
            self.quic_port,
            configuration=configuration,
        ) as protocol:
            self.quic_protocol = protocol
            logger.info("QUIC connection established")

            # Setup UDP socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Start packet processing tasks
            self.running = True
            tasks = [
                asyncio.create_task(self._receive_packets()),
                asyncio.create_task(self._forward_packets()),
            ]

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled, shutting down")
            finally:
                self.running = False
                if self.udp_socket:
                    self.udp_socket.close()

    async def _receive_packets(self):
        logger.info(f"Starting to receive packets from stream {self.stream_id}")
        buffer = bytearray()

        while self.running:
            try:
                # Wait for stream data
                waiter = self.quic_protocol._loop.create_future()
                self.quic_protocol._stream_data_received_waiters[self.stream_id] = (
                    waiter
                )

                await asyncio.wait_for(waiter, timeout=1.0)

                # Read available data from stream
                data = self.quic_protocol._quic.receive_stream_data(
                    self.stream_id,
                    self.packet_class.PACKET_SIZE * 10,  # Read multiple packets at once
                )

                if data:
                    buffer.extend(data)

                    # Process complete packets from buffer
                    while len(buffer) >= self.packet_class.PACKET_SIZE:
                        packet_data = bytes(buffer[: self.packet_class.PACKET_SIZE])
                        buffer = buffer[self.packet_class.PACKET_SIZE :]

                        try:
                            packet = self.packet_class.from_bytes(packet_data)
                            await self.packet_queue.put(packet)
                            logger.debug(
                                f"Received packet with timestamp {packet.timestamp}"
                            )
                        except ValueError as e:
                            logger.error(f"Failed to parse packet: {e}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error receiving packets: {e}")
                await asyncio.sleep(0.1)

    async def _forward_packets(self):
        logger.info(
            f"Starting to forward packets to UDP {self.udp_host}:{self.udp_port}"
        )
        packets_forwarded = 0

        while self.running:
            try:
                # Get packet from queue with timeout
                packet = await asyncio.wait_for(self.packet_queue.get(), timeout=1.0)

                # Convert to bytes and send via UDP
                packet_bytes = packet.to_bytes()
                self.udp_socket.sendto(packet_bytes, (self.udp_host, self.udp_port))

                packets_forwarded += 1
                if packets_forwarded % 100 == 0:
                    logger.info(f"Forwarded {packets_forwarded} packets")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error forwarding packet: {e}")

    async def stop(self):
        logger.info("Stopping VelodyneLidar")
        self.running = False

    async def send_as_quic_packet(self, packet: PacketType) -> None:
        """Send a Velodyne packet to the QUIC server

        Args:
            packet: The packet to send to the QUIC server

        Raises:
            RuntimeError: If QUIC connection is not established
        """
        if not self.quic_protocol:
            raise RuntimeError("QUIC connection not established. Call connect() first.")

        try:
            # Convert packet to bytes
            packet_bytes = packet.to_bytes()

            # Send data on the configured stream
            self.quic_protocol._quic.send_stream_data(
                self.stream_id, packet_bytes, end_stream=False
            )

            # Transmit any pending data
            self.quic_protocol.transmit()

            logger.debug(
                f"Sent packet with timestamp {packet.timestamp} via QUIC stream {self.stream_id}"
            )

        except Exception as e:
            logger.error(f"Failed to send packet via QUIC: {e}")
            raise

    @abstractmethod
    def parse_packet(self, data: bytes) -> PacketType:
        """Parse raw bytes into a packet object

        Args:
            data: Raw packet bytes

        Returns:
            Parsed packet object
        """
        pass

    @abstractmethod
    def validate_packet(self, data: bytes) -> bool:
        """Validate if the data represents a valid packet

        Args:
            data: Raw packet bytes

        Returns:
            True if valid packet, False otherwise
        """
        pass


def main():
    import argparse
    from .vlp16 import VLP16, VLP16Config

    parser = argparse.ArgumentParser(description="Velodyne QUIC to UDP forwarder")
    parser.add_argument("--quic-host", default="localhost", help="QUIC server hostname")
    parser.add_argument("--quic-port", type=int, default=4433, help="QUIC server port")
    parser.add_argument(
        "--udp-host", default="127.0.0.1", help="UDP destination hostname"
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=VLP16Config.udp_port,
        help="UDP destination port",
    )
    parser.add_argument(
        "--stream-id", type=int, default=0, help="QUIC stream ID to read from"
    )
    parser.add_argument(
        "--model",
        default="vlp16",
        choices=["vlp16"],
        help="Velodyne model type (default: vlp16)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def run():
        # Create VLP16 instance
        lidar = VLP16(
            quic_host=args.quic_host,
            quic_port=args.quic_port,
            udp_host=args.udp_host,
            udp_port=args.udp_port,
            stream_id=args.stream_id,
        )

        logger.info("Using VLP-16 packet format")

        try:
            await lidar.connect()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await lidar.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
