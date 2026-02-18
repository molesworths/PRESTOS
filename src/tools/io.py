import numpy as np
import subprocess
import os
import sys
import json
import time
import socket
import stat
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tempfile import TemporaryDirectory
import paramiko
import logging

# Configure logging for platform operations
logger = logging.getLogger(__name__)

def isfloat(x):
    return isinstance(x, (np.floating, float))


def isint(x):
    return isinstance(x, (np.integer, int))


def isnum(x):
    return isinstance(x, (np.floating, float)) or isinstance(x, (np.integer, int))


def islistarray(x):
    return type(x) in [list, np.ndarray]


def isAnyNan(x):
    try:
        aux = len(x)
    except:
        aux = None

    if aux is None:
        isnan = np.isnan(x)
    else:
        isnan = False
        for j in range(aux):
            isnan = isnan or np.isnan(x[j]).any()

    return isnan

def clipstr(txt, chars=40):
    if not isinstance(txt, str):
        txt = f"{txt}"
    return f"{'...' if len(txt) > chars else ''}{txt[-chars:]}" if txt is not None else None


def get_root():
    """Get the root directory of the SMARTS solver standalone package."""
    import os
    from pathlib import Path

    return Path(os.path.dirname(os.path.abspath(__file__))).parents[3]

# ============================================================================
# Platform Infrastructure for PRESTOS Module Execution
# ============================================================================

@dataclass
class PlatformSpec:
    """Specification for a compute platform (machine + hardware).
    
    Attributes
    ----------
    name : str
        Unique identifier for the platform (e.g., 'local', 'engaging_cluster')
    machine : str
        'local' for local machine, or hostname for remote execution
    username : str
        Remote username (defaults to current user for local machine)
    scratch : str
        Working directory path (local path or remote path)
    n_cpu : int
        Number of CPU cores available per node
    n_gpu : int
        Number of GPUs available per node (0 if none)
    n_ram_gb : float
        RAM available in GB
    modules : str
        Shell commands to load environment (e.g., 'module load gcc/11.2.0')
    ssh_identity : str
        Path to SSH private key for remote access
    ssh_tunnel : Optional[str]
        Jump host for SSH tunneling (if behind firewall)
    ssh_port : int
        SSH port (default 22)
    scheduler : str
        Job scheduler type: 'slurm', 'none' (for local execution)
    slurm_partition : str
        SLURM partition name (if using SLURM)
    slurm_qos : str
        SLURM quality-of-service (if using SLURM)
    gacode_root : str
        Path to GACODE installation (sets GACODE_ROOT)
    """
    name: str
    machine: str = "local"
    username: str = ""
    scratch: str = "."
    n_cpu: int = 1
    n_gpu: int = 0
    n_ram_gb: float = 8.0
    modules: str = ""
    ssh_identity: str = ""
    ssh_tunnel: Optional[str] = None
    ssh_port: int = 22
    scheduler: str = "none"
    slurm_partition: str = "default"
    slurm_qos: str = "default"
    gacode_root: str = ""

    def __post_init__(self):
        if not self.username:
            self.username = os.getenv("USER", "user")
        self.scratch = os.path.expanduser(os.path.expandvars(self.scratch))
        if self.gacode_root:
            self.gacode_root = os.path.expanduser(os.path.expandvars(self.gacode_root))
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PlatformSpec":
        """Create PlatformSpec from configuration dictionary."""
        spec = cls(
            name=config.get("name", "platform"),
            machine=config.get("machine", "local"),
            username=config.get("username", os.getenv("USER", "user")),
            scratch=config.get("scratch", "."),
            n_cpu=config.get("n_cpu", config.get("cores_per_node", 1)),
            n_gpu=config.get("n_gpu", config.get("gpus_per_node", 0)),
            n_ram_gb=config.get("n_ram_gb", 8.0),
            modules=config.get("modules", ""),
            ssh_identity=config.get("ssh_identity", ""),
            ssh_tunnel=config.get("ssh_tunnel", None),
            ssh_port=config.get("ssh_port", 22),
            scheduler=config.get("scheduler", "none"),
            slurm_partition=config.get("slurm_partition", "default"),
            slurm_qos=config.get("slurm_qos", "default"),
            gacode_root=config.get("gacode_root", ""),
        )
        return spec
    
    def is_local(self) -> bool:
        """Check if this is local execution (not remote)."""
        return self.machine == "local" or self.machine == "localhost"
    
    def get_scratch_path(self) -> Path:
        """Get scratch directory as Path object."""
        return Path(self.scratch).expanduser()


class CommandExecutor:
    """Execute commands locally or on remote systems via SSH."""
    
    def __init__(self, platform: PlatformSpec):
        self.platform = platform
        self._ssh_client = None
    
    def _get_ssh_client(self) -> paramiko.SSHClient:
        """Get or create SSH client for remote execution."""
        if self._ssh_client is None:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if self.platform.ssh_tunnel:
                # SSH tunneling: local -> tunnel -> target
                tunnel_client = paramiko.SSHClient()
                tunnel_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                tunnel_client.connect(
                    self.platform.ssh_tunnel,
                    username=self.platform.username,
                    key_filename=self.platform.ssh_identity or None,
                    port=self.platform.ssh_port,
                )
                # Create port forward through tunnel (simplified: use direct connection)
                # Note: Full tunneling would require paramiko's Transport layer setup
                logger.warning("SSH tunneling not fully implemented; using direct connection")
                self._ssh_client.connect(
                    self.platform.machine,
                    username=self.platform.username,
                    key_filename=self.platform.ssh_identity or None,
                    port=self.platform.ssh_port,
                )
            else:
                # Direct SSH connection
                self._ssh_client.connect(
                    self.platform.machine,
                    username=self.platform.username,
                    key_filename=self.platform.ssh_identity or None,
                    port=self.platform.ssh_port,
                )
        return self._ssh_client
    
    def execute(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command locally or remotely.
        
        Parameters
        ----------
        command : str
            Command to execute
        cwd : Path, optional
            Working directory (local only)
        timeout : int, optional
            Timeout in seconds
        check : bool
            If True, raise exception on non-zero exit code
        
        Returns
        -------
        returncode : int
        stdout : str
        stderr : str
        """
        if self.platform.is_local():
            return self._execute_local(command, cwd, timeout, check)
        else:
            return self._execute_remote(command, cwd, timeout, check)
    
    def _execute_local(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command locally."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if check and result.returncode != 0:
                raise RuntimeError(f"Command failed with code {result.returncode}: {result.stderr}")
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")
    
    def _execute_remote(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command remotely via SSH."""
        ssh = self._get_ssh_client()
        
        # Prepend environment setup and cd if needed
        full_command = command
        if self.platform.modules:
            full_command = f"{self.platform.modules} && {full_command}"
        if cwd:
            full_command = f"cd {cwd} && {full_command}"
        
        try:
            stdin, stdout, stderr = ssh.exec_command(full_command, timeout=timeout)
            stdout_str = stdout.read().decode('utf-8', errors='ignore')
            stderr_str = stderr.read().decode('utf-8', errors='ignore')
            returncode = stdout.channel.recv_exit_status()
            
            if check and returncode != 0:
                raise RuntimeError(f"Remote command failed with code {returncode}: {stderr_str}")
            
            return returncode, stdout_str, stderr_str
        except socket.timeout:
            raise TimeoutError(f"Remote command timed out after {timeout}s: {command}")
    
    def close(self):
        """Close SSH connection."""
        if self._ssh_client:
            self._ssh_client.close()
            self._ssh_client = None


class FileManager:
    """Manage file transfers and staging between local and remote systems."""
    
    def __init__(self, platform: PlatformSpec):
        self.platform = platform
        self._sftp_client = None
    
    def _get_sftp_client(self) -> paramiko.SFTPClient:
        """Get or create SFTP client for file transfers."""
        if self._sftp_client is None:
            from paramiko import Transport
            executor = CommandExecutor(self.platform)
            ssh = executor._get_ssh_client()
            self._sftp_client = paramiko.SFTPClient.from_transport(ssh.get_transport())
        return self._sftp_client
    
    def upload_file(self, local_path: Path, remote_path: Path) -> None:
        """Upload single file to remote system."""
        if self.platform.is_local():
            # Local copy
            import shutil
            shutil.copy2(str(local_path), str(remote_path))
        else:
            # SFTP upload
            sftp = self._get_sftp_client()
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.put(str(local_path), str(remote_path))
        logger.info(f"Uploaded {local_path} -> {remote_path}")
    
    def download_file(self, remote_path: Path, local_path: Path) -> None:
        """Download single file from remote system."""
        if self.platform.is_local():
            # Local copy
            import shutil
            shutil.copy2(str(remote_path), str(local_path))
        else:
            # SFTP download
            sftp = self._get_sftp_client()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(str(remote_path), str(local_path))
        logger.info(f"Downloaded {remote_path} -> {local_path}")
    
    def upload_directory(self, local_dir: Path, remote_dir: Path) -> None:
        """Upload entire directory to remote system."""
        if self.platform.is_local():
            # Local copy
            import shutil
            shutil.copytree(str(local_dir), str(remote_dir), dirs_exist_ok=True)
        else:
            # SFTP recursive upload
            sftp = self._get_sftp_client()
            self._sftp_mkdir_p(sftp, str(remote_dir))
            for item in local_dir.rglob("*"):
                rel_path = item.relative_to(local_dir)
                remote_item = remote_dir / rel_path
                if item.is_file():
                    self._sftp_mkdir_p(sftp, str(remote_item.parent))
                    sftp.put(str(item), str(remote_item))
        logger.info(f"Uploaded directory {local_dir} -> {remote_dir}")
    
    def download_directory(self, remote_dir: Path, local_dir: Path) -> None:
        """Download entire directory from remote system."""
        if self.platform.is_local():
            # Local copy
            import shutil
            shutil.copytree(str(remote_dir), str(local_dir), dirs_exist_ok=True)
        else:
            # SFTP recursive download
            sftp = self._get_sftp_client()
            local_dir.mkdir(parents=True, exist_ok=True)
            self._sftp_walk(sftp, str(remote_dir), str(local_dir))
        logger.info(f"Downloaded directory {remote_dir} -> {local_dir}")
    
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
            executor = CommandExecutor(self.platform)
            executor.execute(f"rm -rf {path}", check=False)
    
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
            
            if item.filename.startswith('.'):
                continue
            
            if stat.S_ISDIR(item.st_mode):
                FileManager._sftp_walk(sftp, remote_item, str(local_item))
            else:
                sftp.get(remote_item, str(local_item))
    
    def close(self):
        """Close SFTP connection."""
        if self._sftp_client:
            self._sftp_client.close()
            self._sftp_client = None


class SLURMJobSubmitter:
    """Generate and submit SLURM batch jobs."""
    
    def __init__(self, platform: PlatformSpec):
        if platform.scheduler != "slurm":
            raise ValueError(f"Platform scheduler is '{platform.scheduler}', not 'slurm'")
        self.platform = platform
        self.executor = CommandExecutor(platform)
    
    def generate_batch_script(
        self,
        command: str,
        job_name: str,
        n_tasks: int = 1,
        cpus_per_task: int = 1,
        walltime_minutes: int = 30,
        output_file: Optional[str] = None,
        error_file: Optional[str] = None,
        modules: Optional[str] = None,
        job_array: Optional[str] = None,
    ) -> str:
        """Generate SLURM batch script.
        
        Parameters
        ----------
        command : str
            Command to execute in the job
        job_name : str
            Name for the SLURM job
        n_tasks : int
            Number of MPI tasks
        cpus_per_task : int
            CPUs per task
        walltime_minutes : int
            Wall time in minutes
        output_file : str, optional
            Output file path
        error_file : str, optional
            Error file path
        modules : str, optional
            Module load commands (overrides platform.modules)
        job_array : str, optional
            Job array specification (e.g., "1-5%2")
        
        Returns
        -------
        script : str
            SLURM batch script content
        """
        env_setup = modules or self.platform.modules
        hours = walltime_minutes // 60
        mins = walltime_minutes % 60
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH --ntasks={n_tasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition={self.platform.slurm_partition}
"""
        if job_array:
            script += f"#SBATCH --array={job_array}\n"
        if output_file:
            script += f"#SBATCH --output={output_file}\n"
        if error_file:
            script += f"#SBATCH --error={error_file}\n"
        
        script += f"\n# Load environment\n{env_setup}\n\n" if env_setup else "\n"
        script += f"# Execute command\n{command}\n"
        
        return script
    
    def submit_job(
        self,
        script_content: str,
        script_path: Path,
        wait: bool = False,
    ) -> str:
        """Submit SLURM job and return job ID.
        
        Parameters
        ----------
        script_content : str
            SLURM batch script content
        script_path : Path
            Path where script will be saved
        wait : bool
            If True, wait for job completion
        
        Returns
        -------
        job_id : str
            SLURM job ID
        """
        # Write script locally or remotely
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.platform.is_local():
            with open(script_path, 'w') as f:
                f.write(script_content)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(script_content)
                temp_path = f.name
            
            fm = FileManager(self.platform)
            fm.upload_file(Path(temp_path), script_path)
            os.remove(temp_path)
        
        # Make executable
        self.executor.execute(f"chmod +x {script_path}", check=False)
        
        # Submit job
        _, stdout, _ = self.executor.execute(
            f"sbatch {script_path}",
            cwd=script_path.parent,
            check=True,
        )
        
        # Parse job ID from sbatch output
        # Output format: "Submitted batch job 12345"
        job_id = stdout.split()[-1].strip()
        logger.info(f"Submitted SLURM job {job_id}")
        
        if wait:
            self._wait_for_job(job_id)
        
        return job_id
    
    def _wait_for_job(self, job_id: str, check_interval: int = 10) -> bool:
        """Wait for SLURM job to complete.
        
        Parameters
        ----------
        job_id : str
            SLURM job ID
        check_interval : int
            Seconds between status checks
        
        Returns
        -------
        success : bool
            True if job completed successfully
        """
        while True:
            _, stdout, _ = self.executor.execute(
                f"squeue -j {job_id} --format='%T'",
                check=False,
            )
            
            status = stdout.strip().split('\n')[-1] if stdout else "UNKNOWN"
            
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                logger.info(f"Job {job_id} status: {status}")
                return status == "COMPLETED"
            
            logger.info(f"Job {job_id} still running (status: {status})")
            time.sleep(check_interval)


class PlatformManager:
    """High-level orchestration for running PRESTOS modules on different platforms."""
    
    def __init__(self, platform: Union[PlatformSpec, Dict[str, Any]]):
        """Initialize platform manager.
        
        Parameters
        ----------
        platform : PlatformSpec or dict
            Platform configuration
        """
        if isinstance(platform, dict):
            platform = PlatformSpec.from_dict(platform)
        self.platform = platform
        self.executor = CommandExecutor(platform)
        self.file_manager = FileManager(platform)
        self.slurm_submitter = None
        if platform.scheduler == "slurm":
            self.slurm_submitter = SLURMJobSubmitter(platform)
    
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
    ) -> None:
        """Stage input files to remote platform.
        
        Parameters
        ----------
        local_dir : Path
            Local directory containing inputs
        remote_dir : Path
            Target directory on platform
        file_patterns : list, optional
            File patterns to copy (default: all)
        """
        self.file_manager.mkdir(remote_dir)
        
        if file_patterns is None:
            # Copy entire directory
            self.file_manager.upload_directory(local_dir, remote_dir)
        else:
            # Copy specific patterns
            for pattern in file_patterns:
                for file_path in local_dir.glob(pattern):
                    rel_path = file_path.relative_to(local_dir)
                    dest_path = remote_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    self.file_manager.upload_file(file_path, dest_path)
    
    def retrieve_outputs(
        self,
        remote_dir: Path,
        local_dir: Path,
        file_patterns: Optional[List[str]] = None,
    ) -> None:
        """Retrieve output files from platform.
        
        Parameters
        ----------
        remote_dir : Path
            Remote directory containing outputs
        local_dir : Path
            Target local directory
        file_patterns : list, optional
            File patterns to copy (default: all)
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        
        if file_patterns is None:
            # Copy entire directory
            self.file_manager.download_directory(remote_dir, local_dir)
        else:
            # Copy specific patterns
            if self.platform.is_local():
                import glob
                for pattern in file_patterns:
                    for file_path in glob.glob(str(remote_dir / pattern)):
                        rel_path = Path(file_path).relative_to(remote_dir)
                        dest_path = local_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        self.file_manager.download_file(Path(file_path), dest_path)
            else:
                # Remote: use find or similar
                for pattern in file_patterns:
                    _, stdout, _ = self.executor.execute(
                        f"find {remote_dir} -name '{pattern}' -type f",
                        check=False,
                    )
                    for file_path in stdout.strip().split('\n'):
                        if file_path:
                            rel_path = Path(file_path).relative_to(remote_dir)
                            dest_path = local_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            self.file_manager.download_file(Path(file_path), dest_path)
    
    def cleanup(self) -> None:
        """Close all connections."""
        self.executor.close()
        self.file_manager.close()


