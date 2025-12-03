"""
Test cases for V4L2 Ansible setup functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from hilspy.setup.v4l2_setup import V4L2Setup


class TestV4L2Setup:
    """Test cases for V4L2Setup class"""

    def test_init_default_playbook_dir(self):
        """Test V4L2Setup initialization with default playbook directory"""
        setup = V4L2Setup()
        assert setup.playbook_dir.name == "playbooks"
        assert setup.playbook_path.name == "setup-v4l2.yml"

    def test_init_custom_playbook_dir(self):
        """Test V4L2Setup initialization with custom playbook directory"""
        custom_dir = Path("/tmp/custom")
        setup = V4L2Setup(playbook_dir=custom_dir)
        assert setup.playbook_dir == custom_dir
        assert setup.playbook_path == custom_dir / "setup-v4l2.yml"

    def test_check_prerequisites_missing_playbook(self):
        """Test prerequisites check when playbook is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup = V4L2Setup(playbook_dir=Path(temp_dir))
            assert not setup.check_prerequisites()

    @patch("os.geteuid")
    def test_check_prerequisites_not_root(self, mock_geteuid):
        """Test prerequisites check when not running as root"""
        mock_geteuid.return_value = 1000  # Non-root user

        # Create a temporary playbook file
        with tempfile.TemporaryDirectory() as temp_dir:
            playbook_dir = Path(temp_dir)
            playbook_path = playbook_dir / "setup-v4l2.yml"
            playbook_path.write_text("---\n- hosts: localhost\n")

            setup = V4L2Setup(playbook_dir=playbook_dir)
            assert not setup.check_prerequisites()

    @patch("os.geteuid")
    def test_check_prerequisites_success(self, mock_geteuid):
        """Test prerequisites check when all conditions are met"""
        mock_geteuid.return_value = 0  # Root user

        # Create a temporary playbook file
        with tempfile.TemporaryDirectory() as temp_dir:
            playbook_dir = Path(temp_dir)
            playbook_path = playbook_dir / "setup-v4l2.yml"
            playbook_path.write_text("---\n- hosts: localhost\n")

            setup = V4L2Setup(playbook_dir=playbook_dir)
            assert setup.check_prerequisites()

    def test_verify_setup_device_exists(self):
        """Test verify_setup when device exists"""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            setup = V4L2Setup()
            assert setup.verify_setup()

    def test_verify_setup_device_missing(self):
        """Test verify_setup when device is missing"""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            setup = V4L2Setup()
            assert not setup.verify_setup()

    @patch("ansible_runner.run")
    @patch("os.geteuid")
    def test_run_setup_success(self, mock_geteuid, mock_ansible_run):
        """Test successful setup run"""
        mock_geteuid.return_value = 0  # Root user
        mock_result = Mock()
        mock_result.status = "successful"
        mock_ansible_run.return_value = mock_result

        # Create a temporary playbook file
        with tempfile.TemporaryDirectory() as temp_dir:
            playbook_dir = Path(temp_dir)
            playbook_path = playbook_dir / "setup-v4l2.yml"
            playbook_path.write_text("---\n- hosts: localhost\n")

            setup = V4L2Setup(playbook_dir=playbook_dir)
            assert setup.run_setup()

    @patch("ansible_runner.run")
    @patch("os.geteuid")
    def test_run_setup_failure(self, mock_geteuid, mock_ansible_run):
        """Test failed setup run"""
        mock_geteuid.return_value = 0  # Root user
        mock_result = Mock()
        mock_result.status = "failed"
        mock_result.stderr = "Error message"
        mock_ansible_run.return_value = mock_result

        # Create a temporary playbook file
        with tempfile.TemporaryDirectory() as temp_dir:
            playbook_dir = Path(temp_dir)
            playbook_path = playbook_dir / "setup-v4l2.yml"
            playbook_path.write_text("---\n- hosts: localhost\n")

            setup = V4L2Setup(playbook_dir=playbook_dir)
            assert not setup.run_setup()


def test_playbook_file_exists():
    """Test that the actual playbook file exists in the project"""
    project_root = Path(__file__).parent.parent
    playbook_path = (
        project_root / "src" / "hilspy" / "setup" / "playbooks" / "setup-v4l2.yml"
    )
    assert playbook_path.exists(), f"Playbook file not found: {playbook_path}"


def test_playbook_has_required_tasks():
    """Test that the playbook contains required tasks"""
    project_root = Path(__file__).parent.parent
    playbook_path = (
        project_root / "src" / "hilspy" / "setup" / "playbooks" / "setup-v4l2.yml"
    )

    content = playbook_path.read_text()

    # Check for key components
    assert "v4l2loopback-dkms" in content
    assert "ffmpeg" in content
    assert "modprobe" in content
    assert "hilspy" in content


def test_setup_script_has_main_function():
    """Test that the setup script can be imported and has main function"""
    from hilspy.setup.v4l2_setup import main, V4L2Setup

    assert callable(main)
    assert V4L2Setup is not None
