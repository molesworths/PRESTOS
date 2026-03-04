from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import resolve_platform_config


@dataclass
class PlatformSpec:
    """Specification for a compute platform (machine + hardware).

    Attributes
    ----------
    name : str
        Unique identifier for the platform (e.g., "local", "engaging_cluster")
    machine : str
        "local" for local machine, or hostname for remote execution
    username : str
        Remote username (defaults to current user for local machine)
    scratch : str
        Working directory path (local path or remote path)
    n_cpu : int
        Number of CPU cores available per node
    n_gpus : int
        Number of GPUs available per node (0 if none)
    n_ram_gb : float
        RAM available in GB
    modules : str
        Shell commands to load environment (e.g., "module load gcc/11.2.0")
    ssh_identity : str
        Path to SSH private key for remote access
    ssh_tunnel : Optional[str]
        Jump host for SSH tunneling (if behind firewall)
    ssh_tunnel_port : int
        SSH port on the tunnel/jump host (if using SSH tunneling, default 22)
    ssh_port : int
        SSH port on the target machine (default 22)
    scheduler : str
        Job scheduler type: "slurm", "none" (for local execution)
    slurm_partition : str
        SLURM partition name (if using SLURM)
    slurm_qos : str
        SLURM quality-of-service (if using SLURM)
    slurm_constraint : str
        SLURM node constraints (e.g., "cascadelake", "gpu")
    mpi_tasks : int
        Number of MPI tasks (maps to GACODE -n flag)
    threads_per_task : int
        Number of OpenMP threads per MPI task (maps to GACODE -nomp flag)
    """

    name: str
    machine: str = "local"
    username: str = ""
    scratch: str = "."
    n_cpu: int = 1
    n_gpus: int = 0
    n_ram_gb: float = 8.0
    modules: str = ""
    ssh_identity: str = ""
    ssh_tunnel: Optional[str] = None
    ssh_tunnel_port: int = 22
    ssh_port: int = 22
    scheduler: str = "none"
    slurm_partition: str = "default"
    slurm_qos: str = "default"
    slurm_constraint: str = ""
    mpi_tasks: int = 1
    threads_per_task: int = 1

    def __post_init__(self) -> None:
        if not self.username:
            self.username = os.getenv("USER", "user")
        self.scratch = os.path.expanduser(os.path.expandvars(self.scratch))

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PlatformSpec":
        """Create PlatformSpec from configuration dictionary.

        Accepts either flat keys (e.g. ``slurm_partition``) or a nested
        ``slurm`` sub-dict (e.g. ``slurm: {partition: short}``) as written
        in run_config.yaml.
        """
        resolved = resolve_platform_config(config)

        slurm_sub = resolved.get("slurm", {}) or {}
        spec = cls(
            name=resolved.get("name", "platform"),
            machine=resolved.get("machine", "local"),
            username=resolved.get("username", os.getenv("USER", "user")),
            scratch=resolved.get("scratch", "."),
            n_cpu=resolved.get("n_cpu", resolved.get("cores_per_node", 1)),
            n_gpus=resolved.get("n_gpus", resolved.get("gpus_per_node", 0)),
            n_ram_gb=resolved.get("n_ram_gb", 8.0),
            modules=resolved.get("modules", ""),
            ssh_identity=resolved.get("ssh_identity", ""),
            ssh_tunnel=resolved.get("ssh_tunnel", None),
            ssh_tunnel_port=resolved.get("ssh_tunnel_port", 22),
            ssh_port=resolved.get("ssh_port", 22),
            scheduler=resolved.get("scheduler", "none"),
            slurm_partition=resolved.get("slurm_partition", slurm_sub.get("partition", "default")),
            slurm_qos=resolved.get("slurm_qos", slurm_sub.get("qos", slurm_sub.get("account", "default"))),
            slurm_constraint=resolved.get("slurm_constraint", slurm_sub.get("constraint", "")),
            mpi_tasks=int(resolved.get("mpi_tasks", resolved.get("n_parallel", 1))),
            threads_per_task=int(resolved.get("threads_per_task", resolved.get("n_threads", 1))),
        )
        return spec

    def is_local(self) -> bool:
        """Check if this is local execution (not remote)."""
        return self.machine == "local" or self.machine == "localhost"

    def get_scratch_path(self) -> Path:
        """Get scratch directory as Path object."""
        return Path(self.scratch).expanduser()
