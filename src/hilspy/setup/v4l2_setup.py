#!/usr/bin/env python3
"""
V4L2 Environment Setup Script for hilspy

This script uses ansible-runner to execute the V4L2 environment setup playbook.
It automates the installation and configuration of v4l2loopback for hilspy camera functionality.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import ansible_runner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class V4L2Setup:
    """V4L2 environment setup using Ansible playbook"""

    def __init__(self, playbook_dir: Optional[Path] = None):
        if playbook_dir is None:
            # Get the directory containing this script
            script_dir = Path(__file__).parent
            self.playbook_dir = script_dir / "playbooks"
        else:
            self.playbook_dir = playbook_dir

        self.playbook_path = self.playbook_dir / "setup-v4l2.yml"

    def check_prerequisites(self) -> bool:
        """Check if required files exist"""
        if not self.playbook_path.exists():
            logger.error(f"Playbook not found: {self.playbook_path}")
            return False

        if os.geteuid() != 0:
            logger.error("This script requires root privileges. Please run with sudo.")
            return False

        return True

    def run_setup(self, verbose: bool = False) -> bool:
        """Execute the V4L2 setup playbook"""
        if not self.check_prerequisites():
            return False

        logger.info("Starting V4L2 environment setup...")
        logger.info(f"Using playbook: {self.playbook_path}")

        try:
            # Run the Ansible playbook
            result = ansible_runner.run(
                playbook=str(self.playbook_path),
                inventory="localhost,",
                verbosity=2 if verbose else 1,
                quiet=not verbose,
                suppress_ansible_output=not verbose,
            )

            if result.status == "successful":
                logger.info("V4L2 setup completed successfully!")
                logger.info(
                    "You can now use /dev/video0 with hilspy camera functionality."
                )
                return True
            else:
                logger.error(f"Setup failed with status: {result.status}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to run setup: {e}")
            return False

    def verify_setup(self) -> bool:
        """Verify that V4L2 device is available"""
        video_device = Path("/dev/video0")
        if video_device.exists():
            logger.info("✓ V4L2 device /dev/video0 is available")
            return True
        else:
            logger.warning("✗ V4L2 device /dev/video0 is not available")
            return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup V4L2 environment for hilspy camera functionality"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the current setup without running playbook",
    )
    parser.add_argument(
        "--playbook-dir", type=Path, help="Custom path to playbook directory"
    )

    args = parser.parse_args()

    setup = V4L2Setup(playbook_dir=args.playbook_dir)

    if args.verify_only:
        success = setup.verify_setup()
        sys.exit(0 if success else 1)

    # Run the setup
    success = setup.run_setup(verbose=args.verbose)

    if success:
        # Verify the setup worked
        setup.verify_setup()
        logger.info("Setup completed. You can now use hilspy camera functionality.")
        sys.exit(0)
    else:
        logger.error("Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
