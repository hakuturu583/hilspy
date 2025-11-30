"""VLP-16 specific implementation for Velodyne LiDAR"""

import struct
import math
from dataclasses import dataclass
from typing import List, ClassVar, Optional
import time

from .velodyne_lidar import VelodynePacket, DataBlock, LaserReturn, VelodyneLidar


@dataclass
class VLP16Config:
    """VLP-16 specific configuration"""

    # Packet structure constants
    PACKET_SIZE: ClassVar[int] = 1206
    BLOCKS_PER_PACKET: ClassVar[int] = 12
    LASERS_PER_BLOCK: ClassVar[int] = 32  # 2 firings * 16 lasers
    CHANNELS: ClassVar[int] = 16

    # Data block constants
    BLOCK_SIZE: ClassVar[int] = 100  # 2 (flag) + 2 (azimuth) + 32*3 (laser data)
    UPPER_BLOCK: ClassVar[int] = 0xEEFF
    LOWER_BLOCK: ClassVar[int] = 0xDDFF

    # Measurement constants
    DISTANCE_RESOLUTION: ClassVar[float] = 0.002  # 2mm
    ROTATION_MAX_UNITS: ClassVar[int] = 36000  # 0.01 degree units

    # Laser elevation angles for VLP-16 (in degrees)
    LASER_ANGLES: ClassVar[List[float]] = [
        -15.0,
        1.0,
        -13.0,
        3.0,
        -11.0,
        5.0,
        -9.0,
        7.0,
        -7.0,
        9.0,
        -5.0,
        11.0,
        -3.0,
        13.0,
        -1.0,
        15.0,
    ]

    # Timing constants
    SINGLE_FIRING_TIME: ClassVar[float] = 55.296  # microseconds
    DOUBLE_FIRING_TIME: ClassVar[float] = 110.592  # microseconds

    # Network defaults
    udp_port: int = 2368
    udp_host: str = "127.0.0.1"

    # Return mode
    return_mode: int = 0x37  # Strongest return
    product_id: int = 0x22  # VLP-16


@dataclass
class VLP16Packet(VelodynePacket):
    """VLP-16 specific packet implementation"""

    pass

    @classmethod
    def create_from_points(
        cls,
        azimuth_start: float,
        azimuth_step: float = 0.2,
        distances: Optional[List[List[float]]] = None,
        intensities: Optional[List[List[int]]] = None,
    ) -> "VLP16Packet":
        """
        Create a VLP16 packet from point cloud data

        Args:
            azimuth_start: Starting azimuth angle in degrees
            azimuth_step: Azimuth step between blocks in degrees
            distances: List of distance arrays for each block (in meters)
            intensities: List of intensity arrays for each block (0-255)
        """
        blocks = []

        for block_idx in range(VLP16Config.BLOCKS_PER_PACKET):
            # Alternate between upper and lower blocks
            flag = (
                VLP16Config.UPPER_BLOCK
                if block_idx % 2 == 0
                else VLP16Config.LOWER_BLOCK
            )

            # Calculate azimuth for this block
            azimuth = azimuth_start + (block_idx * azimuth_step)
            if azimuth >= 360.0:
                azimuth -= 360.0

            laser_returns = []
            for laser_idx in range(VLP16Config.LASERS_PER_BLOCK):
                if (
                    distances
                    and block_idx < len(distances)
                    and laser_idx < len(distances[block_idx])
                ):
                    # Convert meters to 2mm units
                    distance = int(
                        distances[block_idx][laser_idx]
                        / VLP16Config.DISTANCE_RESOLUTION
                    )
                else:
                    distance = 0

                if (
                    intensities
                    and block_idx < len(intensities)
                    and laser_idx < len(intensities[block_idx])
                ):
                    intensity = intensities[block_idx][laser_idx]
                else:
                    intensity = 0

                laser_returns.append(LaserReturn(distance, intensity))

            blocks.append(DataBlock(flag, azimuth, laser_returns))

        # Generate timestamp (microseconds since top of hour)
        timestamp = int(time.time() * 1000000) % (3600 * 1000000)

        # Factory bytes: return mode and product ID
        factory = struct.pack("BB", VLP16Config.return_mode, VLP16Config.product_id)

        return cls(blocks=blocks, timestamp=timestamp, factory=factory)

    @classmethod
    def validate_packet(cls, data: bytes) -> bool:
        """
        Validate if the data represents a valid VLP-16 packet

        Args:
            data: Raw packet bytes

        Returns:
            True if valid VLP-16 packet, False otherwise
        """
        if len(data) != VLP16Config.PACKET_SIZE:
            return False

        # Check block flags
        offset = 0
        for i in range(VLP16Config.BLOCKS_PER_PACKET):
            flag = struct.unpack("<H", data[offset : offset + 2])[0]
            expected = (
                VLP16Config.UPPER_BLOCK if i % 2 == 0 else VLP16Config.LOWER_BLOCK
            )
            if flag != expected:
                return False
            offset += VLP16Config.BLOCK_SIZE

        return True

    def get_point_cloud(self) -> List[tuple]:
        """
        Convert packet to point cloud (x, y, z, intensity) format

        Returns:
            List of tuples (x, y, z, intensity) in meters
        """
        points = []

        for block in self.blocks:
            azimuth_rad = block.azimuth * 0.017453293  # Convert to radians

            for laser_idx in range(VLP16Config.CHANNELS):
                # Each block has two firings
                for firing in range(2):
                    idx = firing * VLP16Config.CHANNELS + laser_idx
                    if idx >= len(block.laser_returns):
                        continue

                    laser_return = block.laser_returns[idx]
                    if laser_return.distance == 0:
                        continue

                    # Convert distance from 2mm units to meters
                    distance = laser_return.distance * VLP16Config.DISTANCE_RESOLUTION

                    # Get elevation angle for this laser
                    elevation = VLP16Config.LASER_ANGLES[laser_idx]
                    elevation_rad = elevation * 0.017453293

                    # Calculate 3D coordinates
                    xy_distance = distance * math.cos(elevation_rad)
                    x = xy_distance * math.sin(azimuth_rad)
                    y = xy_distance * math.cos(azimuth_rad)
                    z = distance * math.sin(elevation_rad)

                    points.append((x, y, z, laser_return.intensity))

        return points


class VLP16(VelodyneLidar[VLP16Packet]):
    """VLP-16 LiDAR interface implementing the VelodyneLidar abstract base class"""

    def __init__(
        self,
        config: Optional[VLP16Config] = None,
        quic_host: str = "localhost",
        quic_port: int = 4433,
        udp_host: str = "127.0.0.1",
        udp_port: int = 2368,
        stream_id: int = 0,
    ):
        self.config = config or VLP16Config()
        super().__init__(
            packet_class=VLP16Packet,
            quic_host=quic_host,
            quic_port=quic_port,
            udp_host=udp_host,
            udp_port=udp_port,
            stream_id=stream_id,
        )

    def create_packet(self, **kwargs) -> VLP16Packet:
        """Create a VLP16 packet with given parameters"""
        return VLP16Packet.create_from_points(**kwargs)

    def parse_packet(self, data: bytes) -> VLP16Packet:
        """Parse raw bytes into a VLP16Packet

        Args:
            data: Raw packet bytes

        Returns:
            Parsed VLP16Packet object
        """
        if not self.validate_packet(data):
            raise ValueError("Invalid VLP-16 packet data")
        packet = VLP16Packet.from_bytes(data)
        # Ensure we return VLP16Packet type
        if not isinstance(packet, VLP16Packet):
            # This should not happen but satisfy mypy
            return VLP16Packet(
                blocks=packet.blocks, timestamp=packet.timestamp, factory=packet.factory
            )
        return packet

    def validate_packet(self, data: bytes) -> bool:
        """Validate if the data represents a valid VLP-16 packet

        Args:
            data: Raw packet bytes

        Returns:
            True if valid VLP-16 packet, False otherwise
        """
        return VLP16Packet.validate_packet(data)

    def get_config(self) -> VLP16Config:
        """Get current VLP-16 configuration"""
        return self.config
