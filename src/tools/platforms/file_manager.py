from __future__ import annotations

import logging
import stat
from pathlib import Path

import paramiko

from .command_executor import CommandExecutor
from .platformspec import PlatformSpec

logger = logging.getLogger(__name__)


class FileManager:
    """Manage file transfers and staging between local and remote systems."""

    def __init__(self, platform: PlatformSpec):
        self.platform = platform
        self._sftp_client = None
        self._executor = None  # Keep executor alive for tunnel connection
        self._closed = False

    def _get_sftp_client(self) -> paramiko.SFTPClient:
        """Get or create SFTP client for file transfers with tunnel support."""
        if self._sftp_client is None:
            self._executor = CommandExecutor(self.platform)
            ssh = self._executor._get_ssh_client()
            transport = ssh.get_transport()

            if transport is None:
                raise RuntimeError("Failed to establish SSH transport for SFTP")

            self._sftp_client = paramiko.SFTPClient.from_transport(transport)
            logger.info("SFTP client connected via %s", self.platform.machine)

        return self._sftp_client

    def upload_file(self, local_path: Path, remote_path: Path) -> None:
        """Upload single file to remote system."""
        if self.platform.is_local():
            import shutil

            shutil.copy2(str(local_path), str(remote_path))
        else:
            sftp = self._get_sftp_client()
            self._sftp_mkdir_p(sftp, str(remote_path.parent))
            sftp.put(str(local_path), str(remote_path))
        logger.info("Uploaded %s -> %s", local_path, remote_path)

    def download_file(self, remote_path: Path, local_path: Path) -> None:
        """Download single file from remote system."""
        if self.platform.is_local():
            import shutil

            shutil.copy2(str(remote_path), str(local_path))
        else:
            sftp = self._get_sftp_client()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(str(remote_path), str(local_path))
        logger.info("Downloaded %s -> %s", remote_path, local_path)

    def upload_directory(self, local_dir: Path, remote_dir: Path) -> None:
        """Upload entire directory to remote system."""
        if self.platform.is_local():
            import shutil

            shutil.copytree(str(local_dir), str(remote_dir), dirs_exist_ok=True)
        else:
            sftp = self._get_sftp_client()
            self._sftp_mkdir_p(sftp, str(remote_dir))
            for item in local_dir.rglob("*"):
                rel_path = item.relative_to(local_dir)
                remote_item = remote_dir / rel_path
                if item.is_file():
                    self._sftp_mkdir_p(sftp, str(remote_item.parent))
                    sftp.put(str(item), str(remote_item))
        logger.info("Uploaded directory %s -> %s", local_dir, remote_dir)

    def download_directory(self, remote_dir: Path, local_dir: Path) -> None:
        """Download entire directory from remote system."""
        if self.platform.is_local():
            import shutil

            shutil.copytree(str(remote_dir), str(local_dir), dirs_exist_ok=True)
        else:
            sftp = self._get_sftp_client()
            local_dir.mkdir(parents=True, exist_ok=True)
            self._sftp_walk(sftp, str(remote_dir), str(local_dir))
        logger.info("Downloaded directory %s -> %s", remote_dir, local_dir)

    def mkdir(self, path: Path) -> None:
        """Create directory locally or remotely."""
        if self.platform.is_local():
            path.mkdir(parents=True, exist_ok=True)
        else:
            sftp = self._get_sftp_client()
            self._sftp_mkdir_p(sftp, str(path))

    def remove_directory(self, path: Path) -> None:
        """Remove directory locally or remotely."""
        if self.platform.is_local():
            import shutil

            shutil.rmtree(str(path), ignore_errors=True)
        else:
            # Reuse executor if we have one, otherwise create and cleanup
            if self._executor:
                self._executor.execute(f"rm -rf {path}", check=False)
            else:
                executor = CommandExecutor(self.platform)
                try:
                    executor.execute(f"rm -rf {path}", check=False)
                finally:
                    executor.close()

    @staticmethod
    def _sftp_mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
        """Create directory recursively via SFTP."""
        try:
            sftp.stat(path)
        except IOError:
            parent = str(Path(path).parent)
            if parent != path:
                FileManager._sftp_mkdir_p(sftp, parent)
            sftp.mkdir(path)

    @staticmethod
    def _sftp_walk(
        sftp: paramiko.SFTPClient,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Recursively download directory via SFTP."""
        local_path_obj = Path(local_path)
        local_path_obj.mkdir(parents=True, exist_ok=True)

        try:
            items = sftp.listdir_attr(remote_path)
        except IOError:
            return

        for item in items:
            remote_item = f"{remote_path}/{item.filename}"
            local_item = local_path_obj / item.filename

            if item.filename.startswith("."):
                continue

            if stat.S_ISDIR(item.st_mode):
                FileManager._sftp_walk(sftp, remote_item, str(local_item))
            else:
                sftp.get(remote_item, str(local_item))

    def close(self) -> None:
        """Close SFTP connection and executor."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._sftp_client:
            try:
                self._sftp_client.close()
            except Exception:
                pass
            self._sftp_client = None

        if self._executor:
            self._executor.close()
            self._executor = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup is called."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Silently fail in destructor to avoid issues during shutdown
