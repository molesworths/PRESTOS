"""
Configuration Example: Using Platform Infrastructure with PRESTOS Workflows

This file demonstrates how to configure platform execution in workflow YAML files.
"""

# Example 1: Workflow with Local Platform Execution
local_workflow_yaml = """
work_dir: ~/prestos_work
state:
  args:
    from_gacode: ~/example/input.gacode

parameters:
  class: parameterizations.PowerLawParameterization
  args:
    n_params: 5

transport:
  class: transport.FingerprintsModel
  args:
    options:
      roa_eval: [0.3, 0.5, 0.7]
      modes: all
  # Platform for transport model execution
  platform:
    machine: local
    scratch: ./transport_scratch
    n_cpu: 4

solver:
  class: solvers.GradientDescentSolver
  args:
    n_iterations: 10
"""

# Example 2: Workflow with Remote Platform Execution
remote_workflow_yaml = """
work_dir: ~/prestos_work
state:
  args:
    from_gacode: ~/example/input.gacode

parameters:
  class: parameterizations.PowerLawParameterization
  args:
    n_params: 5

transport:
  class: transport.FingerprintsModel
  args:
    options:
      roa_eval: [0.3, 0.5, 0.7]
      modes: all
  # Platform for transport model execution (remote)
  platform:
    machine: remote.example.com
    username: user
    scratch: /work/user/prestos
    n_cpu: 16
    ssh_identity: ~/.ssh/remote_key
    modules: "module load python/3.9"

solver:
  class: solvers.GradientDescentSolver
  args:
    n_iterations: 10
"""

# Example 3: Workflow with SLURM Cluster Execution
slurm_workflow_yaml = """
work_dir: ~/prestos_work
state:
  args:
    from_gacode: ~/example/input.gacode

parameters:
  class: parameterizations.PowerLawParameterization
  args:
    n_params: 5

transport:
  class: transport.FingerprintsModel
  args:
    options:
      roa_eval: [0.3, 0.5, 0.7]
      modes: all
  # Platform for transport model execution (SLURM cluster)
  platform:
    machine: hpc.cluster.org
    username: user
    scratch: /home/user/work
    n_cpu: 64
    n_gpu: 4
    modules: "module load gcc/11.2.0 && module load openmpi/4.1.0"
    ssh_identity: ~/.ssh/cluster_key
    scheduler: slurm
    slurm_partition: gpu

solver:
  class: solvers.GradientDescentSolver
  args:
    n_iterations: 10
"""

# Example 4: Integrated Python Script
integrated_python_example = """
import yaml
import json
from pathlib import Path
from interfaces import gacode
from state import PlasmaState
from transport import FingerprintsModel
from tools.io import PlatformManager, PlatformSpec

def run_with_platform(workflow_file, platform_config_file=None):
    '''Run transport model using platform configuration from YAML.'''
    
    # Load workflow configuration
    with open(workflow_file) as f:
        workflow_config = yaml.safe_load(f)
    
    # Build state
    gacode_path = workflow_config['state']['args']['from_gacode']
    gc = gacode(filepath=Path(gacode_path).expanduser())
    state = PlasmaState.from_gacode(gc)
    state.process(gc)
    
    # Build transport model
    transport_config = workflow_config.get('transport', {})
    transport = FingerprintsModel(
        options=transport_config['args']['options']
    )
    
    # Get platform configuration
    platform_config = transport_config.get('platform', {})
    if isinstance(platform_config, dict):
        platform = PlatformManager(platform_config)
    else:
        raise ValueError("Platform must be dict")
    
    # Run on platform
    output_dict, std_dict = transport.run_on_platform(
        state,
        platform,
        work_dir=Path(workflow_config['work_dir']) / 'transport_work',
    )
    
    return output_dict, std_dict

# Usage
if __name__ == '__main__':
    # Run with local platform
    output, std = run_with_platform('workflow_local.yaml')
    print(f"Transport fluxes: {output}")
"""

# Example 5: Platform Configuration File (JSON)
platform_config_json = """
{
  "local_machine": {
    "name": "local_development",
    "machine": "local",
    "scratch": "~/prestos_work/scratch",
    "n_cpu": 16,
    "n_gpu": 0,
    "n_ram_gb": 64.0,
    "scheduler": "none"
  },

  "engaging_cluster": {
    "name": "engaging_mit",
    "machine": "engaging.mit.edu",
    "username": "user",
    "scratch": "/home/user/scratch",
    "n_cpu": 40,
    "n_gpu": 0,
    "n_ram_gb": 192.0,
    "modules": "module load gcc/11.2.0 && module load openmpi/4.1.0",
    "ssh_identity": "~/.ssh/engaging_key",
    "scheduler": "slurm",
    "slurm_partition": "default",
    "slurm_qos": "default"
  },

  "perlmutter_nersc": {
    "name": "perlmutter",
    "machine": "perlmutter.nersc.gov",
    "username": "user",
    "scratch": "/pscratch/sd/u/user",
    "n_cpu": 128,
    "n_gpu": 4,
    "n_ram_gb": 512.0,
    "modules": "module load PrgEnv-gnu && module load gpu && module load cudatoolkit",
    "ssh_identity": "~/.ssh/nersc_key",
    "scheduler": "slurm",
    "slurm_partition": "gpu"
  },

  "local_cluster_hpc": {
    "name": "local_hpc",
    "machine": "hpc.internal.org",
    "username": "user",
    "ssh_tunnel": "gateway.external.org",
    "scratch": "/work/user/prestos",
    "n_cpu": 64,
    "n_gpu": 8,
    "n_ram_gb": 256.0,
    "modules": "module load intel/2021 && module load mkl",
    "ssh_identity": "~/.ssh/id_rsa",
    "scheduler": "slurm",
    "slurm_partition": "gpu"
  }
}
"""

# Example 6: Advanced Usage with Multiple Platforms
advanced_multi_platform = """
from tools.io import PlatformManager, PlatformSpec
import json
from pathlib import Path

def run_parameter_scan_distributed(state_params, platform_specs):
    '''Run parameter scan across multiple platforms.'''
    
    transport = FingerprintsModel(options={...})
    results = {}
    
    # Distribute parameter sets across platforms
    params_per_platform = len(state_params) // len(platform_specs)
    
    for platform_idx, platform_config in enumerate(platform_specs):
        platform = PlatformManager(platform_config)
        platform_name = platform_config.get('name', f'platform_{platform_idx}')
        
        start_idx = platform_idx * params_per_platform
        end_idx = start_idx + params_per_platform
        
        platform_params = state_params[start_idx:end_idx]
        
        results[platform_name] = []
        
        for param_idx, params in enumerate(platform_params):
            # Build state from parameters
            state = build_state_from_params(params)
            
            # Run on platform
            output, std = transport.run_on_platform(
                state,
                platform,
                model_name=f'param_set_{platform_idx}_{param_idx}',
            )
            
            results[platform_name].append({
                'params': params,
                'output': output,
                'std': std,
            })
        
        platform.cleanup()
    
    return results

# Load platforms from configuration
with open('platforms.json') as f:
    platforms_config = json.load(f)

# Get subset of platforms for this run
active_platforms = [
    platforms_config['local_machine'],
    platforms_config['engaging_cluster'],
    platforms_config['perlmutter_nersc'],
]

# Run distributed parameter scan
results = run_parameter_scan_distributed(
    state_params=[...],  # List of parameter sets
    platform_specs=active_platforms,
)
"""

# Print examples
if __name__ == '__main__':
    print("Example 1: Local Workflow")
    print("="*70)
    print(local_workflow_yaml)
    
    print("\nExample 2: Remote Workflow")
    print("="*70)
    print(remote_workflow_yaml)
    
    print("\nExample 3: SLURM Cluster Workflow")
    print("="*70)
    print(slurm_workflow_yaml)
    
    print("\nExample 4: Integrated Python Script")
    print("="*70)
    print(integrated_python_example)
    
    print("\nExample 5: Platform Configuration (JSON)")
    print("="*70)
    print(platform_config_json)
    
    print("\nExample 6: Multi-Platform Parameter Scan")
    print("="*70)
    print(advanced_multi_platform)
