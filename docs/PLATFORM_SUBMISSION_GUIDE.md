# PRESTOS Platform Submission Guide

## Overview

The PRESTOS platform infrastructure enables running PRESTOS modules (such as `transport.FingerprintsModel`) on different compute platforms:

- **Local execution** – Direct execution on your machine
- **Remote SSH execution** – On remote machines via SSH
- **SLURM clusters** – HPC environments with SLURM job scheduling
- **Multiple platforms** – Configure different platforms for rapid switching

## Architecture

The platform infrastructure consists of four core components in `src/tools/io.py`:

1. **PlatformSpec** – Describes platform configuration (machine, hardware, scheduler)
2. **CommandExecutor** – Executes commands locally or remotely via SSH
3. **FileManager** – Manages file transfers (SFTP for remote, local copy for local)
4. **SLURMJobSubmitter** – Generates and submits SLURM batch jobs
5. **PlatformManager** – High-level orchestration combining all above components

## Configuration

### Platform Specification

Define platforms in a JSON configuration file:

```json
{
  "local_machine": {
    "machine": "local",
    "scratch": "~/scratch/prestos",
    "n_cpu": 16,
    "n_gpu": 0,
    "n_ram_gb": 64.0,
    "scheduler": "none"
  },
  "hpc_cluster": {
    "machine": "cluster.example.com",
    "username": "user",
    "scratch": "/work/user/prestos",
    "n_cpu": 64,
    "n_gpu": 4,
    "n_ram_gb": 256.0,
    "modules": "module load gcc/11.2.0 && module load openmpi/4.1.0",
    "ssh_identity": "~/.ssh/cluster_key",
    "scheduler": "slurm",
    "slurm_partition": "gpu"
  },
  "remote_machine": {
    "machine": "gateway.example.com",
    "username": "user",
    "scratch": "/home/user/work",
    "ssh_tunnel": "jump.example.com",
    "ssh_identity": "~/.ssh/id_rsa",
    "n_cpu": 8,
    "scheduler": "none"
  }
}
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `machine` | Hostname or "local" for local execution |
| `username` | Remote username (defaults to current user) |
| `scratch` | Working directory on platform (can use `~`) |
| `n_cpu` | Number of CPU cores available |
| `n_gpu` | Number of GPUs available |
| `n_ram_gb` | Available RAM in GB |
| `modules` | Shell commands to load environment |
| `ssh_identity` | Path to SSH private key |
| `ssh_tunnel` | Jump host for SSH tunneling (if behind firewall) |
| `ssh_port` | SSH port (default 22) |
| `scheduler` | Job scheduler: "slurm" or "none" |
| `slurm_partition` | SLURM partition name (e.g., "gpu", "cpu") |
| `slurm_qos` | SLURM quality-of-service level |

## Usage Examples

### 1. Direct Local Execution

```python
from transport import FingerprintsModel
from state import PlasmaState
from tools.io import PlatformManager, PlatformSpec

# Create transport model
transport_model = FingerprintsModel(
    options={
        'roa_eval': [0.3, 0.5, 0.7],
        'modes': 'all',
    }
)

# Load plasma state
state = PlasmaState.from_gacode(gc)
state.process(gc)

# Option A: Direct evaluation (no platform)
output_dict, std_dict = transport_model._evaluate_single(state)

# Option B: Evaluate on local platform
platform_config = PlatformSpec(
    name="local",
    machine="local",
    scratch="./work",
)
output_dict, std_dict = transport_model.run_on_platform(
    state,
    platform_config,
    work_dir="./transport_work",
)
```

### 2. Remote SSH Execution

```python
from tools.io import PlatformManager, PlatformSpec

# Configure remote platform
platform_config = PlatformSpec(
    name="remote_server",
    machine="server.example.com",
    username="user",
    scratch="/home/user/prestos_work",
    ssh_identity="~/.ssh/remote_key",
)

# Run on remote platform
output_dict, std_dict = transport_model.run_on_platform(
    state,
    platform_config,
    work_dir="./transport_work",
    cleanup=True,  # Remove remote scratch after completion
)
```

### 3. SLURM Cluster Execution

```python
from tools.io import PlatformManager, PlatformSpec

# Configure SLURM cluster
platform_config = PlatformSpec.from_dict({
    "name": "engaging_cluster",
    "machine": "engaging.mit.edu",
    "username": "user",
    "scratch": "/home/user/scratch",
    "modules": "module load gcc/11.2.0 && module load openmpi/4.1.0",
    "ssh_identity": "~/.ssh/engaging_key",
    "scheduler": "slurm",
    "slurm_partition": "default",
    "n_cpu": 40,
    "n_gpu": 0,
})

output_dict, std_dict = transport_model.run_on_platform(
    state,
    platform_config,
    work_dir="./transport_work",
)
```

### 4. Load Configuration from JSON

```python
import json
from tools.io import PlatformManager, PlatformSpec

# Load platform config from JSON file
with open('platforms.json', 'r') as f:
    platforms_config = json.load(f)

# Get specific platform
platform_spec = PlatformSpec.from_dict(platforms_config['hpc_cluster'])

# Run model
output_dict, std_dict = transport_model.run_on_platform(state, platform_spec)
```

### 5. Batch Processing Multiple States

```python
from tools.io import PlatformManager, PlatformSpec

# Define platform
platform_spec = PlatformSpec.from_dict(platforms_config['hpc_cluster'])

# Batch evaluate multiple states
states = [state1, state2, state3]
results = []

for i, state in enumerate(states):
    output_dict, std_dict = transport_model.run_on_platform(
        state,
        platform_spec,
        model_name=f"transport_run_{i}",
        cleanup=True,
    )
    results.append((output_dict, std_dict))
```

### 6. Direct Platform Manager Usage

For more control, use `PlatformManager` directly:

```python
from tools.io import PlatformManager, PlatformSpec
from pathlib import Path

# Create platform manager
platform = PlatformManager(platform_spec)

# Stage inputs
local_input_dir = Path("./inputs")
remote_input_dir = platform.platform.get_scratch_path() / "run1"
platform.stage_inputs(local_input_dir, remote_input_dir)

# Execute command
returncode, stdout, stderr = platform.run_command(
    "python compute_transport.py input.pkl",
    cwd=remote_input_dir,
)

# Retrieve outputs
local_output_dir = Path("./outputs")
platform.retrieve_outputs(remote_input_dir, local_output_dir)

# Cleanup
platform.cleanup()
```

## Advanced Features

### SSH Tunneling (Jump Hosts)

For clusters behind firewalls:

```python
platform_config = PlatformSpec.from_dict({
    "machine": "cluster.internal.com",
    "username": "user",
    "ssh_tunnel": "gateway.external.com",  # Jump host
    "ssh_identity": "~/.ssh/id_rsa",
    "scratch": "/work/user",
})

# Platform manager handles tunneling transparently
transport_model.run_on_platform(state, platform_config)
```

### SLURM Job Arrays (Bulk Submissions)

For batch parameter scans:

```python
from tools.io import SLURMJobSubmitter

submitter = SLURMJobSubmitter(platform_spec)

# Generate batch script for parameter sweep
script = submitter.generate_batch_script(
    command="python param_scan.py",
    job_name="param_sweep",
    n_tasks=1,
    cpus_per_task=4,
    walltime_minutes=60,
    job_array="1-100%10",  # 100 jobs, max 10 concurrent
)

job_id = submitter.submit_job(
    script,
    Path("/remote/work/run.batch"),
    wait=False,  # Don't block, return job ID
)
```

### File Staging with Patterns

```python
from pathlib import Path

# Stage only specific file patterns
platform.stage_inputs(
    local_input_dir,
    remote_input_dir,
    file_patterns=["*.pkl", "config*.json"],
)

# Retrieve specific outputs
platform.retrieve_outputs(
    remote_output_dir,
    local_output_dir,
    file_patterns=["output*.pkl", "*.log"],
)
```

## Integration with Solvers

Use platform execution within a solver workflow:

```python
from solvers import SolverBase
from transport import FingerprintsModel
from tools.io import PlatformManager, PlatformSpec

class TransportOptimizationSolver(SolverBase):
    def __init__(self, platform_config=None):
        self.transport = FingerprintsModel(options={...})
        self.platform_config = platform_config
    
    def evaluate_transport(self, state):
        if self.platform_config:
            # Run on configured platform
            output_dict, std_dict = self.transport.run_on_platform(
                state,
                self.platform_config,
            )
        else:
            # Run locally
            output_dict, std_dict = self.transport._evaluate_single(state)
        
        return output_dict, std_dict

# Use in solver
solver = TransportOptimizationSolver(
    platform_config=platforms_config['hpc_cluster']
)
```

## Error Handling and Recovery

```python
from tools.io import PlatformManager, CommandExecutor

try:
    output_dict, std_dict = transport_model.run_on_platform(
        state,
        platform_spec,
        cleanup=False,  # Keep remote files if error
    )
except Exception as e:
    print(f"Model execution failed: {e}")
    
    # Retrieve error logs for debugging
    platform = PlatformManager(platform_spec)
    try:
        platform.retrieve_outputs(
            remote_work_dir,
            Path("./error_logs"),
            file_patterns=["*.log", "error*"],
        )
    except:
        pass
    finally:
        platform.cleanup()
```

## Performance Considerations

### Local vs Remote Execution

- **Local**: Fast for small jobs, no network overhead, limited by local hardware
- **Remote**: Scales to larger jobs on HPC systems, network I/O overhead for small jobs

### File Transfer Optimization

- Use `file_patterns` to transfer only necessary files
- Compress large inputs/outputs if possible
- Consider incremental updates for iterative workflows

### SLURM Job Configuration

```python
# Quick jobs (< 5 min)
slurm_setup = {
    'n_tasks': 1,
    'cpus_per_task': 4,
    'walltime_minutes': 10,
}

# Long-running jobs (> 1 hour)
slurm_setup = {
    'n_tasks': 1,
    'cpus_per_task': 64,
    'walltime_minutes': 300,  # 5 hours
    'job_array': '1-50%5',  # Batch in groups
}
```

## Troubleshooting

### SSH Connection Refused
- Verify SSH key path: `ssh_identity` must point to valid private key
- Check remote machine hostname and port
- Test manually: `ssh -i ~/.ssh/key user@host`

### File Transfer Failures
- Ensure remote scratch directory exists and is writable
- Check disk space: `df -h /scratch`
- Verify SSH file permissions: `chmod 600 ~/.ssh/*`

### SLURM Job Not Running
- Check partition availability: `sinfo`
- Verify resource requests don't exceed partition limits
- Check job status: `squeue -u $USER`

### Module Loading Errors
- Test module commands manually on remote system
- Use full paths if modules command not found
- Check module environment variables are set correctly

## API Reference

### PlatformSpec

```python
from tools.io import PlatformSpec

# Create platform specification
platform = PlatformSpec(
    name="my_platform",
    machine="host.example.com",
    username="user",
    scratch="/work/user",
    n_cpu=16,
    n_gpu=2,
    modules="module load python/3.9",
    scheduler="slurm",
)

# Properties
platform.is_local()                    # bool
platform.get_scratch_path()           # Path
```

### PlatformManager

```python
from tools.io import PlatformManager

manager = PlatformManager(platform)

# Methods
manager.run_command(command, cwd, timeout)
manager.stage_inputs(local_dir, remote_dir, file_patterns)
manager.retrieve_outputs(remote_dir, local_dir, file_patterns)
manager.cleanup()
```

### TransportBase

```python
from transport import TransportBase

model = TransportBase(options={...})

# Methods
model.evaluate(state)                              # Local execution
model.run_on_platform(state, platform, work_dir)  # Platform execution
```

## See Also

- [Simulation_Submission_Guide.md](PORTALS_new_ref/Simulation_Submission_Guide.md) – MITIM-fusion reference
- [src/transport.py](src/transport.py) – Transport model implementations
- [src/tools/io.py](src/tools/io.py) – Platform infrastructure implementation
