#!/usr/bin/env python3
"""Test script for VLP16 implementation"""

from hilspy.velodyne import VLP16, VLP16Packet, VLP16Config


def test_vlp16_config():
    """Test VLP16Config constants"""
    config = VLP16Config()

    assert config.PACKET_SIZE == 1206
    assert config.BLOCKS_PER_PACKET == 12
    assert config.LASERS_PER_BLOCK == 32
    assert config.CHANNELS == 16
    assert len(config.LASER_ANGLES) == 16

    print("✓ VLP16Config constants validated")


def test_vlp16_packet_creation():
    """Test creating VLP16 packets"""
    vlp16 = VLP16()

    # Create a packet with sample data
    packet = vlp16.create_packet(
        azimuth_start=90.0,
        azimuth_step=0.2,
        distances=[[2.0] * 32 for _ in range(12)],  # 2 meters for all points
        intensities=[[100] * 32 for _ in range(12)],  # Intensity 100 for all
    )

    # Check packet structure
    assert len(packet.blocks) == 12
    assert packet.blocks[0].azimuth == 90.0
    assert packet.blocks[1].azimuth == 90.2

    # Check laser returns
    assert packet.blocks[0].laser_returns[0].distance == 1000  # 2m = 1000 * 2mm
    assert packet.blocks[0].laser_returns[0].intensity == 100

    print("✓ VLP16 packet created successfully")
    return packet


def test_vlp16_serialization():
    """Test packet serialization and validation"""
    vlp16 = VLP16()

    # Create and serialize packet
    packet = vlp16.create_packet(azimuth_start=45.0)
    data = packet.to_bytes()

    # Validate packet
    assert VLP16Packet.validate_packet(data)
    assert len(data) == VLP16Config.PACKET_SIZE

    # Parse back
    parsed = vlp16.parse_packet(data)
    assert parsed.blocks[0].azimuth == 45.0

    print("✓ VLP16 packet serialization/deserialization works")


def test_vlp16_invalid_packet():
    """Test invalid packet handling"""
    vlp16 = VLP16()

    # Test with wrong size
    try:
        vlp16.parse_packet(b"short")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with wrong flags
    bad_data = bytearray(1206)
    bad_data[0:2] = b"\x00\x00"  # Wrong flag

    try:
        vlp16.parse_packet(bytes(bad_data))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ Invalid packet handling works correctly")


if __name__ == "__main__":
    print("Testing VLP16 implementation...\n")

    test_vlp16_config()
    packet = test_vlp16_packet_creation()
    test_vlp16_serialization()
    test_vlp16_invalid_packet()

    print("\n✅ All VLP16 tests passed!")
    print("\nSample packet info:")
    print("  - Model: VLP-16")
    print(f"  - Blocks: {len(packet.blocks)}")
    print(f"  - First azimuth: {packet.blocks[0].azimuth}°")
    print(f"  - Product ID: 0x{packet.factory[1]:02x}")
