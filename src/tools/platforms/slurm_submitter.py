from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from .command_executor import CommandExecutor
from .file_manager import FileManager
from .platformspec import PlatformSpec
from .utils import _build_module_setup

logger = logging.getLogger(__name__)


class SLURMJobSubmitter:
    """Generate and submit SLURM batch jobs.

    Omega cluster partition limits (from omega_guide):
    - ops-rerun: ops only
    - preemptable: no resource limits (can be preempted)
    - short: 30 min max, 32 CPU max
    - medium: 24 hours max, 16 CPU max (default)
    - long: 7 days max, 10 CPU max
    - gpus: GPUs available with --gres=gpu:N
    - comsol: single node, 7 days max
    - ga-ird: GA funded only, 512 CPU max, 7 days max
    - ga-preempt: GA funded only (can be preempted)
    """

    PARTITION_LIMITS = {
        "short": {"walltime_min": 30, "cpus_max": 32},
        "medium": {"walltime_min": 24 * 60, "cpus_max": 16},
        "long": {"walltime_min": 7 * 24 * 60, "cpus_max": 10},
        "preemptable": {"walltime_min": None, "cpus_max": None},
        "ga-ird": {"walltime_min": 7 * 24 * 60, "cpus_max": 512},
        "ga-preempt": {"walltime_min": None, "cpus_max": None},
        "gpus": {"walltime_min": 24 * 60, "cpus_max": 16},
        "comsol": {"walltime_min": 7 * 24 * 60, "cpus_max": None},
    }

    def __init__(self, platform: PlatformSpec):
        if platform.scheduler != "slurm":
            raise ValueError(f"Platform scheduler is '{platform.scheduler}', not 'slurm'")
        self.platform = platform
        self.executor = CommandExecutor(platform)

    def validate_partition_resources(
        self,
        walltime_minutes: int,
        cpus_total: int,
        partition: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Validate resource request against partition limits."""
        partition = partition or self.platform.slurm_partition
        warnings = ""
        errors = ""

        limits = self.PARTITION_LIMITS.get(partition, {})
        if not limits:
            warnings += f"Warning: Unknown partition '{partition}'. Cannot validate resource limits.\n"
            return warnings, errors

        max_walltime = limits.get("walltime_min")
        max_cpus = limits.get("cpus_max")

        if max_walltime is not None and walltime_minutes > max_walltime:
            error_msg = (
                f"Walltime {walltime_minutes} min exceeds {partition} partition limit "
                f"({max_walltime} min = {max_walltime // 60}h {max_walltime % 60}m)"
            )
            errors += error_msg + "\n"
            logger.error(error_msg)

        if max_cpus is not None and cpus_total > max_cpus:
            error_msg = (
                f"Total CPUs {cpus_total} exceed {partition} partition limit ({max_cpus} CPUs)"
            )
            errors += error_msg + "\n"
            logger.error(error_msg)

        return warnings, errors

    def generate_batch_script(
        self,
        command: str,
        job_name: str,
        n_tasks: int = 1,
        cpus_per_task: int = 1,
        walltime_minutes: int = 30,
        memory_gb: Optional[int] = None,
        n_gpus: int = 0,
        output_file: Optional[str] = None,
        error_file: Optional[str] = None,
        modules: Optional[str] = None,
        job_array: Optional[str] = None,
        shell_pre_commands: Optional[str] = None,
        shell_post_commands: Optional[str] = None,
    ) -> str:
        """Generate SLURM batch script."""
        cpus_total = n_tasks * cpus_per_task
        warnings, errors = self.validate_partition_resources(
            walltime_minutes, cpus_total, self.platform.slurm_partition
        )

        if errors:
            raise ValueError(
                "SLURM resource request exceeds partition limits:\n"
                f"{errors}"
                f"Partition: {self.platform.slurm_partition}\n"
                f"Requested: {walltime_minutes} min, {cpus_total} CPUs"
            )

        if warnings:
            logger.warning("SLURM resource warnings:\n%s", warnings)

        env_setup = modules or self.platform.modules
        if env_setup:
            env_setup = _build_module_setup(env_setup)
        hours = walltime_minutes // 60
        mins = walltime_minutes % 60

        script = f"""#!/usr/bin/env bash
            #SBATCH --job-name={job_name}
            #SBATCH --time={hours:02d}:{mins:02d}:00
            #SBATCH --ntasks={n_tasks}
            #SBATCH --cpus-per-task={cpus_per_task}
            #SBATCH --partition={self.platform.slurm_partition}
            """

        mem = memory_gb if memory_gb is not None else int(self.platform.n_ram_gb)
        if mem > 0:
            script += f"#SBATCH --mem={mem}G\n"

        if self.platform.slurm_qos and self.platform.slurm_qos != "default":
            script += f"#SBATCH --qos={self.platform.slurm_qos}\n"

        if self.platform.slurm_constraint:
            script += f"#SBATCH --constraint={self.platform.slurm_constraint}\n"

        if n_gpus > 0:
            script += f"#SBATCH --gres=gpu:{n_gpus}\n"

        if job_array:
            script += f"#SBATCH --array={job_array}\n"

        if output_file:
            script += f"#SBATCH --output={output_file}\n"
        if error_file:
            script += f"#SBATCH --error={error_file}\n"

        script += "\nexport SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK\n"
        script += "echo \"Submitting SLURM job $SLURM_JOBID in $HOSTNAME (host: $SLURM_SUBMIT_HOST)\"\n"
        script += "echo \"Nodes have $SLURM_CPUS_ON_NODE cores and $SLURM_JOB_NUM_NODES node(s) were allocated for this job\"\n"
        script += "echo \"Each of the $SLURM_NTASKS tasks allocated will run with $SLURM_CPUS_PER_TASK cores, allocating $SRUN_CPUS_PER_TASK CPUs per srun\"\n"
        script += "echo \"***********************************************************************************************\"\n"
        script += "echo \"\"\n"
        script += "\nshopt -s expand_aliases\n"

        script += "\n# Load environment\n"
        script += "source $HOME/.bashrc 2>/dev/null || true\n"
        if env_setup:
            script += f"{env_setup}\n"

        if shell_pre_commands:
            script += f"\n# Pre-execution commands\n{shell_pre_commands}\n"

        script += f"\n# Execute command\n{command}\n"

        if shell_post_commands:
            script += f"\n# Post-execution commands\n{shell_post_commands}\n"

        return script

    def submit_job(
        self,
        script_content: str,
        script_path: Path,
        wait: bool = False,
    ) -> str:
        """Submit SLURM job and return job ID."""
        if self.platform.is_local():
            script_path.parent.mkdir(parents=True, exist_ok=True)
            with open(script_path, "w") as f:
                f.write(script_content)
        else:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(script_content)
                temp_path = f.name

            fm = FileManager(self.platform)
            fm.mkdir(script_path.parent)
            fm.upload_file(Path(temp_path), script_path)
            os.remove(temp_path)

        self.executor.execute(f"chmod +x {script_path}", check=False)

        _, stdout, _ = self.executor.execute(
            f"sbatch {script_path}",
            cwd=script_path.parent,
            check=True,
        )

        job_id = stdout.split()[-1].strip()
        logger.info("Submitted SLURM job %s", job_id)

        if wait:
            self._wait_for_job(job_id)

        return job_id

    def _wait_for_job(self, job_id: str, check_interval: int = 10) -> bool:
        """Wait for SLURM job to complete."""
        while True:
            _, stdout, _ = self.executor.execute(
                f"squeue -j {job_id} --format='%T'",
                check=False,
            )

            status = stdout.strip().split("\n")[-1] if stdout else "UNKNOWN"

            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                logger.info("Job %s status: %s", job_id, status)
                return status == "COMPLETED"

            logger.info("Job %s still running (status: %s)", job_id, status)
            time.sleep(check_interval)
