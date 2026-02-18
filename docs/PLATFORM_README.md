# PRESTOS Platform Infrastructure

## Quick Start

The platform infrastructure allows running PRESTOS modules (like `transport.FingerprintsModel`) on different compute platforms:

```python
from transport import FingerprintsModel
from tools.io import PlatformSpec

# Create model
model = FingerprintsModel(options={'roa_eval': [0.3, 0.5, 0.7]})

# Option 1: Local execution (unchanged)
output, std = model._evaluate_single(state)

# Option 2: Local through platform manager
platform = PlatformSpec(name="local", machine="local")
output, std = model.run_on_platform(state, platform)

# Option 3: Remote execution
platform = PlatformSpec.from_dict({
    "machine": "cluster.example.com",
    "username": "user",
    "scheduler": "slurm",
})
output, std = model.run_on_platform(state, platform)
```

## What's New

### Core Components (src/tools/io.py)

| Component | Purpose |
|-----------|---------|
| `PlatformSpec` | Describes target platform (machine, hardware, scheduler) |
| `CommandExecutor` | Executes commands locally or via SSH |
| `FileManager` | Transfers files via copy or SFTP |
| `SLURMJobSubmitter` | Generates and submits SLURM batch jobs |
| `PlatformManager` | High-level orchestration (all components) |

### Transport Integration (src/transport.py)

| Method | Purpose |
|--------|---------|
| `TransportBase.run_on_platform()` | Run model on platform with automatic serialization |

## Documentation

| File | Purpose |
|------|---------|
| **PLATFORM_SUBMISSION_GUIDE.md** | Comprehensive user guide with examples |
| **IMPLEMENTATION_SUMMARY.md** | Technical architecture and design |
| **example/platform_examples.py** | 6 runnable working examples |
| **example/platform_config_examples.py** | Configuration patterns and templates |

## Configuration

### Simple Configuration (Dict/JSON)

```python
# Create platform from dict
platform_config = {
    "machine": "cluster.example.com",
    "username": "user",
    "scratch": "/work/user",
    "n_cpu": 16,
    "scheduler": "slurm",
    "slurm_partition": "gpu",
}
platform = PlatformManager(platform_config)
```

### Configuration File (JSON)

```json
{
  "local_machine": {
    "machine": "local",
    "n_cpu": 16,
    "scheduler": "none"
  },
  "remote_cluster": {
    "machine": "cluster.org",
    "username": "user",
    "scheduler": "slurm",
    "n_cpu": 64,
    "n_gpu": 4
  }
}
```

## Platform Types

| Platform | Machine | Scheduler | Use Case |
|----------|---------|-----------|----------|
| **Local** | "local" | "none" | Development, testing |
| **Remote SSH** | hostname | "none" | Single-node remote execution |
| **SLURM Cluster** | hostname | "slurm" | Large parameter scans, HPC |

## Usage Patterns

### Pattern 1: Direct Local Execution
```python
# No platform needed - direct evaluation
output, std = model._evaluate_single(state)
```

### Pattern 2: Platform-Based Local
```python
# Same result but through platform interface
platform = PlatformSpec(name="local", machine="local")
output, std = model.run_on_platform(state, platform)
```

### Pattern 3: Remote SSH
```python
platform = PlatformSpec(
    machine="remote.org",
    username="user",
    scratch="/home/user/work",
)
output, std = model.run_on_platform(state, platform)
```

### Pattern 4: SLURM Cluster
```python
platform = PlatformSpec(
    machine="hpc.org",
    scheduler="slurm",
    slurm_partition="gpu",
    n_cpu=64,
)
output, std = model.run_on_platform(state, platform)
```

### Pattern 5: Batch Processing
```python
# Run multiple states on same platform
for i, state in enumerate(states):
    output, std = model.run_on_platform(
        state,
        platform,
        model_name=f"run_{i}",
    )
```

## Advanced Features

### SSH Tunneling (Firewall-Behind Clusters)
```python
platform = PlatformSpec.from_dict({
    "machine": "internal.cluster.org",
    "ssh_tunnel": "gateway.external.org",
    "username": "user",
})
```

### File Pattern Matching
```python
# Only transfer specific files
platform.stage_inputs(
    input_dir,
    remote_dir,
    file_patterns=["*.pkl", "config*.json"],
)
```

### SLURM Job Arrays (Bulk Submissions)
```python
# Configure for 100 parallel parameter sets
submitter = SLURMJobSubmitter(platform)
script = submitter.generate_batch_script(
    command="python compute.py",
    job_array="1-100%10",  # 100 jobs, max 10 concurrent
)
job_id = submitter.submit_job(script, path, wait=False)
```

## Testing

Run examples to verify installation:

```bash
cd /path/to/PRESTOS
python example/platform_examples.py
```

This runs 6 examples:
1. Direct local execution
2. Local platform execution
3. Remote platform config (reference)
4. SLURM platform config (reference)
5. Batch processing
6. Direct platform manager operations

## Key Features

✅ **Unified Interface** - Same code works on local/remote/HPC  
✅ **Configuration-Driven** - Switch platforms with config only  
✅ **Backward Compatible** - Existing code works unchanged  
✅ **Error Handling** - Timeouts, retries, cleanup  
✅ **Secure** - SSH keys, tunneling support  
✅ **Flexible** - Local copy or SFTP, SLURM or none  
✅ **Well Documented** - 4 guide files + examples  

## Architecture Overview

```
User Code
    ↓
TransportBase.run_on_platform(state, platform)
    ├─ Serialize state + model → pickle
    ├─ Transfer to platform (via FileManager)
    ├─ Execute on platform (via CommandExecutor)
    │  └─ For SLURM: submit batch job (via SLURMJobSubmitter)
    ├─ Retrieve results (via FileManager)
    ├─ Deserialize results ← pickle
    └─ Return (output_dict, std_dict)
```

## Installation

### Prerequisites
```bash
pip install paramiko  # For SSH/SFTP support
```

### Verify Installation
```python
from tools.io import PlatformManager, PlatformSpec
print("Platform infrastructure ready!")
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | - | Platform identifier |
| `machine` | str | "local" | Hostname or "local" |
| `username` | str | $USER | Remote username |
| `scratch` | str | "." | Working directory |
| `n_cpu` | int | 1 | CPU cores available |
| `n_gpu` | int | 0 | GPUs available |
| `n_ram_gb` | float | 8.0 | RAM in GB |
| `modules` | str | "" | Environment setup commands |
| `ssh_identity` | str | "" | Path to SSH private key |
| `ssh_tunnel` | str | None | Jump host for tunneling |
| `ssh_port` | int | 22 | SSH port |
| `scheduler` | str | "none" | "slurm" or "none" |
| `slurm_partition` | str | "default" | SLURM partition |
| `slurm_qos` | str | "default" | SLURM QoS level |

## Troubleshooting

### SSH Connection Failed
- Check hostname: `ssh user@hostname`
- Verify key: `ssh -i ~/.ssh/key user@hostname`
- Check port: `ssh -p 22 user@hostname`

### File Transfer Failed
- Verify disk space: `df -h /work/user`
- Check permissions: `chmod 755 /work/user`
- Test SFTP: `sftp -i ~/.ssh/key user@hostname`

### SLURM Job Failed
- Check partition: `sinfo`
- Check job: `squeue -u $USER`
- View errors: `tail -f slurm-*.out`

## Integration Examples

### With Solvers
```python
from solvers import SolverBase

class OptimizationSolver(SolverBase):
    def __init__(self, platform_config=None):
        self.transport = FingerprintsModel(...)
        self.platform = PlatformManager(platform_config) if platform_config else None
    
    def evaluate(self, state):
        if self.platform:
            return self.transport.run_on_platform(state, self.platform)
        else:
            return self.transport._evaluate_single(state)
```

### With Workflows (YAML)
```yaml
transport:
  class: transport.FingerprintsModel
  platform:
    machine: cluster.example.com
    scheduler: slurm
```

## API Quick Reference

```python
# Create platform
platform = PlatformSpec(machine="local")
platform = PlatformSpec.from_dict(config_dict)
platform = PlatformManager(config)

# Run model on platform
output, std = model.run_on_platform(state, platform)

# Advanced operations
platform.run_command(command, cwd, timeout)
platform.stage_inputs(local_dir, remote_dir)
platform.retrieve_outputs(remote_dir, local_dir)
platform.cleanup()
```

## See Also

- [PLATFORM_SUBMISSION_GUIDE.md](PLATFORM_SUBMISSION_GUIDE.md) - Full user guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [example/platform_examples.py](example/platform_examples.py) - Working examples
- [example/platform_config_examples.py](example/platform_config_examples.py) - Configuration examples

## Support

For issues or questions:
1. Check [PLATFORM_SUBMISSION_GUIDE.md](PLATFORM_SUBMISSION_GUIDE.md) - Troubleshooting section
2. Review [example/platform_examples.py](example/platform_examples.py) - Working examples
3. Read docstrings: `help(PlatformManager)`, `help(TransportBase.run_on_platform)`

---

**Status**: Production Ready ✅  
**Version**: 1.0  
**Dependencies**: paramiko  
**Backward Compatible**: Yes  
