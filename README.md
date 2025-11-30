# hilspy

[![CI](https://github.com/hakuturu583/hilspy/actions/workflows/ci.yml/badge.svg)](https://github.com/hakuturu583/hilspy/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/pypi/pyversions/hilspy)](https://pypi.org/project/hilspy/)
[![License](https://img.shields.io/github/license/hakuturu583/hilspy)](LICENSE)

A Python package for working with Velodyne LiDAR data, providing QUIC-based communication and packet processing capabilities.

## Features

- Abstract base class for Velodyne LiDAR devices using ABC and Generic types
- VLP-16 specific implementation with proper type constraints
- QUIC client support for bidirectional communication
- UDP forwarding capabilities
- Point cloud conversion utilities

## Installation

```bash
# Install in development mode
uv pip install -e .
```

## Usage

### Basic VLP-16 Usage

```python
import asyncio
from hilspy.velodyne.vlp16 import VLP16, VLP16Config

async def main():
    # Create VLP-16 instance with custom configuration
    config = VLP16Config(udp_port=2368, udp_host="127.0.0.1")
    vlp16 = VLP16(
        config=config,
        quic_host="localhost",
        quic_port=4433,
        stream_id=0
    )

    # Connect to QUIC server
    await vlp16.connect()

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating and Sending Packets

```python
import asyncio
from hilspy.velodyne.vlp16 import VLP16

async def send_example():
    vlp16 = VLP16()
    await vlp16.connect()

    # Create a packet with sample data
    packet = vlp16.create_packet(
        azimuth_start=0.0,
        azimuth_step=0.2,
        distances=[[1.5, 2.0, 3.0] * 16 for _ in range(12)],  # Sample distances
        intensities=[[100, 150, 200] * 16 for _ in range(12)]  # Sample intensities
    )

    # Send packet to QUIC server
    await vlp16.send_as_quic_packet(packet)

    await vlp16.stop()

asyncio.run(send_example())
```

### Processing Raw Packet Data

```python
from hilspy.velodyne.vlp16 import VLP16

def process_packet_data():
    vlp16 = VLP16()

    # Example raw packet bytes (1206 bytes for VLP-16)
    raw_data = b"..." # Your raw packet data here

    # Validate and parse packet
    if vlp16.validate_packet(raw_data):
        packet = vlp16.parse_packet(raw_data)

        # Convert to point cloud
        points = packet.get_point_cloud()
        for x, y, z, intensity in points:
            print(f"Point: ({x:.3f}, {y:.3f}, {z:.3f}), Intensity: {intensity}")
    else:
        print("Invalid packet data")

process_packet_data()
```

### Using as QUIC to UDP Forwarder

```bash
# Run the built-in forwarder
python -m hilspy.velodyne.velodyne_lidar \
    --quic-host localhost \
    --quic-port 4433 \
    --udp-host 127.0.0.1 \
    --udp-port 2368 \
    --stream-id 0 \
    --verbose
```

### Custom Implementation Example

```python
from hilspy.velodyne.velodyne_lidar import VelodyneLidar
from hilspy.velodyne.vlp16 import VLP16Packet

class CustomVelodyneLidar(VelodyneLidar[VLP16Packet]):
    """Custom implementation of VelodyneLidar"""

    def __init__(self):
        super().__init__(
            packet_class=VLP16Packet,
            quic_host="custom.server.com",
            quic_port=5433
        )

    def parse_packet(self, data: bytes) -> VLP16Packet:
        """Custom packet parsing logic"""
        # Add custom validation or preprocessing here
        return VLP16Packet.from_bytes(data)

    def validate_packet(self, data: bytes) -> bool:
        """Custom packet validation logic"""
        return len(data) == 1206 and self._custom_validation(data)

    def _custom_validation(self, data: bytes) -> bool:
        # Your custom validation logic
        return True

# Usage
async def custom_example():
    lidar = CustomVelodyneLidar()
    await lidar.connect()

    # Create and send a packet
    packet = VLP16Packet.create_from_points(azimuth_start=45.0)
    await lidar.send_as_quic_packet(packet)

    await lidar.stop()
```

## VLP-16 Configuration

The VLP-16 configuration includes:

- **16 laser channels** with predefined elevation angles
- **Packet size**: 1206 bytes
- **12 data blocks** per packet
- **Distance resolution**: 2mm
- **Rotation resolution**: 0.01 degrees

```python
from hilspy.velodyne.vlp16 import VLP16Config

# Access configuration constants
print(f"Laser angles: {VLP16Config.LASER_ANGLES}")
print(f"Packet size: {VLP16Config.PACKET_SIZE}")
print(f"Distance resolution: {VLP16Config.DISTANCE_RESOLUTION}m")
```

## Error Handling

```python
import asyncio
from hilspy.velodyne.vlp16 import VLP16

async def error_handling_example():
    vlp16 = VLP16()

    try:
        # This will raise RuntimeError since connection isn't established
        packet = vlp16.create_packet()
        await vlp16.send_as_quic_packet(packet)
    except RuntimeError as e:
        print(f"Connection error: {e}")

    try:
        await vlp16.connect()
        # Now sending should work
        packet = vlp16.create_packet()
        await vlp16.send_as_quic_packet(packet)
    except Exception as e:
        print(f"Send error: {e}")
    finally:
        await vlp16.stop()

asyncio.run(error_handling_example())
```

## Development

### Running Tests

```bash
# Run tests with uv
uv run pytest test/
```

### Type Checking

```bash
# Install mypy and run type checking
uv pip install mypy
uv run mypy src/hilspy/velodyne/ --ignore-missing-imports
```

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
# Install and run pre-commit
uv pip install pre-commit
pre-commit run --all-files
```

## Architecture

- `VelodyneLidar[PacketType]`: Abstract base class with Generic type constraints
- `VLP16`: Concrete implementation for VLP-16 sensors
- `VelodynePacket`: Base packet structure
- `VLP16Packet`: VLP-16 specific packet with point cloud conversion

## License

This project is licensed under the terms specified in the `pyproject.toml` file.
