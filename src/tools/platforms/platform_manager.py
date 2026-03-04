from __future__ import annotations

import datetime
import logging
import signal
import atexit
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .command_executor import CommandExecutor
from .file_manager import FileManager
from .platformspec import PlatformSpec
from .slurm_submitter import SLURMJobSubmitter

logger = logging.getLogger(__name__)


class PlatformManager:
    """High-level orchestration for running PRESTOS modules on different platforms."""

    # Class-level registry of all active instances for signal handling
    _active_instances: List["PlatformManager"] = []
    _signal_handlers_registered = False

    def __init__(self, platform: Union[PlatformSpec, Dict[str, Any]]):
        """Initialize platform manager."""
        if isinstance(platform, dict):
            platform = PlatformSpec.from_dict(platform)
        self.platform = platform

        print("\n[Platform Configuration]")
        if not platform.is_local():
            print(f"  Machine: {platform.machine}")
            print(f"  Username: {platform.username}")
            print(f"  SSH Port: {platform.ssh_port}")
            if platform.ssh_tunnel:
                print(f"  SSH Tunnel: {platform.ssh_tunnel}:{platform.ssh_tunnel_port}")
            print(f"  SSH Identity: {platform.ssh_identity if platform.ssh_identity else '(not specified)'}")
            print(f"  Scratch: {platform.scratch}")
            print(f"  Scheduler: {platform.scheduler}")
            if platform.scheduler == "slurm":
                print(f"  SLURM Partition: {platform.slurm_partition}")
            print(f"  MPI tasks: {platform.mpi_tasks}  Threads/task: {platform.threads_per_task}")
        else:
            print("  Local execution mode")
        print()

        self.executor = CommandExecutor(platform)
        self.file_manager = FileManager(platform)
        self.slurm_submitter = SLURMJobSubmitter(platform) if platform.scheduler == "slurm" else None

        # Register signal handlers once for all instances
        PlatformManager._register_signal_handlers()
        
        # Add this instance to active instances for signal handler cleanup
        PlatformManager._active_instances.append(self)

    def run_command(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command on platform."""
        return self.executor.execute(command, cwd, timeout, check)

    def stage_inputs(
        self,
        local_dir: Path,
        remote_dir: Path,
        file_patterns: Optional[List[str]] = None,
        folder_patterns: Optional[List[str]] = None,
        specific_files: Optional[List[Path]] = None,
        specific_folders: Optional[List[Path]] = None,
    ) -> None:
        """Stage input files and folders to remote platform."""
        self.file_manager.mkdir(remote_dir)

        if all(x is None for x in [file_patterns, folder_patterns, specific_files, specific_folders]):
            self.file_manager.upload_directory(local_dir, remote_dir)
            return

        if specific_folders:
            for folder_path in specific_folders:
                if not folder_path.is_absolute():
                    folder_path = local_dir / folder_path
                rel_path = folder_path.relative_to(local_dir)
                dest_path = remote_dir / rel_path
                self.file_manager.mkdir(dest_path)

        if folder_patterns:
            for pattern in folder_patterns:
                for folder_path in local_dir.glob(pattern):
                    if folder_path.is_dir():
                        rel_path = folder_path.relative_to(local_dir)
                        dest_path = remote_dir / rel_path
                        self.file_manager.mkdir(dest_path)

        if specific_files:
            for file_path in specific_files:
                if not file_path.is_absolute():
                    file_path = local_dir / file_path
                rel_path = file_path.relative_to(local_dir)
                dest_path = remote_dir / rel_path
                self.file_manager.mkdir(dest_path.parent)
                self.file_manager.upload_file(file_path, dest_path)

        if file_patterns:
            for pattern in file_patterns:
                for file_path in local_dir.glob(pattern):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(local_dir)
                        dest_path = remote_dir / rel_path
                        self.file_manager.mkdir(dest_path.parent)
                        self.file_manager.upload_file(file_path, dest_path)

    def retrieve_outputs(
        self,
        remote_dir: Path,
        local_dir: Path,
        file_patterns: Optional[List[str]] = None,
    ) -> None:
        """Retrieve output files from platform."""
        local_dir.mkdir(parents=True, exist_ok=True)

        if file_patterns is None:
            self.file_manager.download_directory(remote_dir, local_dir)
        else:
            if self.platform.is_local():
                import glob

                for pattern in file_patterns:
                    search_pattern = str(remote_dir / "**" / pattern)
                    for file_path_str in glob.glob(search_pattern, recursive=True):
                        file_path = Path(file_path_str)
                        rel_path = file_path.relative_to(remote_dir)
                        dest_path = local_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        self.file_manager.download_file(file_path, dest_path)
            else:
                for pattern in file_patterns:
                    _, stdout, _ = self.executor.execute(
                        f"find {remote_dir} -name '{pattern}' -type f",
                        check=False,
                    )
                    for file_path_str in stdout.strip().split("\n"):
                        if file_path_str:
                            file_path = Path(file_path_str)
                            try:
                                rel_path = file_path.relative_to(remote_dir)
                            except ValueError:
                                rel_path = Path(file_path.name)
                            dest_path = local_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            self.file_manager.download_file(file_path, dest_path)

    def wait_for_outputs(
        self,
        remote_dir: Path,
        file_patterns: Optional[List[str]],
        timeout: int = 3600,
        poll_interval: int = 15,
    ) -> bool:
        """Wait for expected output files to appear on the platform."""
        if self.platform.is_local() or not file_patterns:
            return True

        start = time.time()
        while time.time() - start < timeout:
            for pattern in file_patterns:
                _, stdout, _ = self.executor.execute(
                    f"find {remote_dir} -name '{pattern}' -type f -size +0c",
                    check=False,
                )
                if stdout.strip():
                    return True
            time.sleep(poll_interval)
        return False

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job on the platform."""
        if self.platform.scheduler != "slurm":
            return {
                "status": "UNKNOWN",
                "job_id": job_id,
                "scheduler": self.platform.scheduler,
            }

        try:
            returncode, stdout, stderr = self.executor.execute(
                f"squeue -j {job_id} --format='%T|%P|%U|%M|%C' --noheader",
                check=False,
                timeout=10,
            )

            if returncode != 0 or not stdout.strip():
                returncode, stdout, stderr = self.executor.execute(
                    f"sacct -j {job_id} --format='State%20,Elapsed,AllocCPUS' --noheader -n",
                    check=False,
                    timeout=10,
                )

                if returncode == 0 and stdout.strip():
                    parts = stdout.strip().split()
                    state = parts[0].strip() if parts else "UNKNOWN"
                    elapsed = parts[1].strip() if len(parts) > 1 else "N/A"
                    cpus = parts[2].strip() if len(parts) > 2 else "N/A"
                    return {
                        "status": state,
                        "job_id": job_id,
                        "time_used": elapsed,
                        "cpus": cpus,
                        "raw_output": stdout.strip(),
                    }
                return {
                    "status": "NOT_FOUND",
                    "job_id": job_id,
                    "raw_output": stdout.strip() if stdout else stderr.strip(),
                }

            parts = stdout.strip().split("|")
            if len(parts) >= 5:
                status, partition, user, time_used, cpus = parts[:5]
                return {
                    "status": status.strip(),
                    "job_id": job_id,
                    "partition": partition.strip(),
                    "user": user.strip(),
                    "time_used": time_used.strip(),
                    "cpus": cpus.strip(),
                    "raw_output": stdout.strip(),
                }
            return {
                "status": "UNKNOWN",
                "job_id": job_id,
                "raw_output": stdout.strip(),
            }

        except Exception as e:
            logger.warning("Error querying job status for %s: %s", job_id, e)
            return {
                "status": "ERROR",
                "job_id": job_id,
                "error": str(e),
            }

    def wait_for_job(
        self,
        job_id: str,
        check_interval: int = 10,
        timeout: Optional[int] = None,
        callback=None,
    ) -> bool:
        """Wait for a job to complete, with optional periodic callbacks."""
        start_time = datetime.datetime.now()

        while True:
            status_dict = self.get_job_status(job_id)
            status = status_dict.get("status", "UNKNOWN")

            if callback:
                try:
                    callback(status_dict)
                except Exception as e:
                    logger.warning("Error in status callback: %s", e)

            if status in ["COMPLETED", "FAILED", "CANCELLED", "NOT_FOUND"]:
                logger.info("Job %s reached terminal state: %s", job_id, status)
                return status == "COMPLETED"

            if timeout is not None:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    logger.warning("Job %s timeout after %ss", job_id, elapsed)
                    return False

            time_used = status_dict.get("time_used", "N/A")
            logger.info("Job %s status: %s (time: %s)", job_id, status, time_used)

            time.sleep(check_interval)

    def cleanup(self) -> None:
        """Close all connections and clean up resources."""
        try:
            self.executor.close()
        except Exception as e:
            logger.warning("Error closing executor: %s", e)

        try:
            self.file_manager.close()
        except Exception as e:
            logger.warning("Error closing file manager: %s", e)
        
        # Remove from active instances list
        try:
            PlatformManager._active_instances.remove(self)
        except ValueError:
            pass  # Already removed

    @classmethod
    def _register_signal_handlers(cls) -> None:
        """Register signal handlers for graceful shutdown on SIGINT/SIGTERM."""
        if cls._signal_handlers_registered:
            return
        
        cls._signal_handlers_registered = True
        
        def _cleanup_all_on_signal(signum, frame):
            """Signal handler to cleanup all active platform connections."""
            logger.warning("Received signal %d, closing connections...", signum)
            for instance in list(cls._active_instances):
                try:
                    instance.cleanup()
                except Exception as e:
                    logger.warning("Error cleaning up instance during signal handling: %s", e)
            # Re-raise the signal for normal Python shutdown
            raise KeyboardInterrupt()
        
        # Register handlers for SIGINT (Ctrl+C) and SIGTERM
        try:
            signal.signal(signal.SIGINT, _cleanup_all_on_signal)
            signal.signal(signal.SIGTERM, _cleanup_all_on_signal)
            logger.debug("Signal handlers registered for graceful shutdown")
        except Exception as e:
            logger.warning("Could not register signal handlers: %s", e)
        
        # Also register atexit handler as backup
        atexit.register(cls._cleanup_all_instances)

    @classmethod
    def _cleanup_all_instances(cls) -> None:
        """Called at program exit to cleanup all remaining connections."""
        for instance in list(cls._active_instances):
            try:
                instance.cleanup()
            except Exception as e:
                logger.warning("Error cleaning up instance at exit: %s", e)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup is called."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently fail in destructor to avoid issues during shutdown
