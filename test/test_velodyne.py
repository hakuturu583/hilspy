#!/usr/bin/env python3
"""Test script for VelodynePacket serialization/deserialization"""

import struct
import time
from hilspy.velodyne import VelodynePacket


def test_packet_serialization():
    """Test creating and serializing a VelodynePacket"""
    # Create a sample packet
    packet = VelodynePacket.create_empty()

    # Modify some values
    packet.timestamp = int(time.time() * 1000000) % (
        3600 * 1000000
    )  # Microseconds in current hour

    # Set some sample data for first block
    packet.blocks[0].azimuth = 45.5  # degrees
    packet.blocks[0].laser_returns[0].distance = 1000  # 2mm units = 2 meters
    packet.blocks[0].laser_returns[0].intensity = 128

    # Serialize to bytes
    data = packet.to_bytes()
    assert len(data) == VelodynePacket.PACKET_SIZE
    print(f"✓ Packet serialized to {len(data)} bytes")

    # Deserialize back
    packet2 = VelodynePacket.from_bytes(data)

    # Verify fields match
    assert packet2.timestamp == packet.timestamp
    assert packet2.blocks[0].azimuth == packet.blocks[0].azimuth
    assert (
        packet2.blocks[0].laser_returns[0].distance
        == packet.blocks[0].laser_returns[0].distance
    )
    assert (
        packet2.blocks[0].laser_returns[0].intensity
        == packet.blocks[0].laser_returns[0].intensity
    )
    print("✓ Packet deserialized correctly")

    return packet


def test_packet_structure():
    """Verify packet structure matches Velodyne spec"""
    packet = VelodynePacket.create_empty()
    data = packet.to_bytes()

    # Check block flags
    offset = 0
    for i in range(VelodynePacket.BLOCKS_PER_PACKET):
        flag = struct.unpack("<H", data[offset : offset + 2])[0]
        expected = 0xEEFF if i % 2 == 0 else 0xDDFF
        assert flag == expected, f"Block {i} has wrong flag: {hex(flag)}"
        offset += 100  # Skip to next block

    print(f"✓ All {VelodynePacket.BLOCKS_PER_PACKET} blocks have correct flags")

    # Verify timestamp and factory bytes positions
    timestamp_offset = VelodynePacket.BLOCKS_PER_PACKET * 100
    timestamp = struct.unpack("<I", data[timestamp_offset : timestamp_offset + 4])[0]
    assert timestamp == 0  # Empty packet has 0 timestamp

    factory_offset = timestamp_offset + 4
    factory = data[factory_offset : factory_offset + 2]
    assert len(factory) == 2

    print("✓ Timestamp and factory bytes at correct positions")


if __name__ == "__main__":
    print("Testing VelodynePacket implementation...")
    print()

    test_packet_structure()
    print()

    packet = test_packet_serialization()
    print()

    print("Sample packet info:")
    print(f"  - Timestamp: {packet.timestamp} μs")
    print(f"  - First block azimuth: {packet.blocks[0].azimuth}°")
    print(
        f"  - First laser distance: {packet.blocks[0].laser_returns[0].distance * 2}mm"
    )
    print(f"  - First laser intensity: {packet.blocks[0].laser_returns[0].intensity}")
    print()

    print("✅ All tests passed!")
