#!/usr/bin/env python3
"""
Example usage of the V4L2 setup automation script.

This demonstrates how to use the hilspy V4L2 setup functionality programmatically.
"""

from hilspy.setup.v4l2_setup import V4L2Setup


def main():
    """Example usage of V4L2Setup class"""
    print("=== hilspy V4L2 Setup Example ===")

    # Create setup instance
    setup = V4L2Setup()

    # First, verify current setup
    print("\n1. Checking current V4L2 setup...")
    setup.verify_setup()

    # Demonstrate how to run setup programmatically
    # Note: This would require sudo privileges
    print("\n2. To run the full setup (requires sudo):")
    print("   setup.run_setup(verbose=True)")
    print("   Or use the CLI command: hilspy-setup-v4l2")

    print("\n3. Available CLI options:")
    print("   hilspy-setup-v4l2 --help              # Show help")
    print("   hilspy-setup-v4l2 --verify-only       # Only check current setup")
    print("   hilspy-setup-v4l2 --verbose           # Run with detailed output")

    print("\n=== Example completed ===")


if __name__ == "__main__":
    main()
