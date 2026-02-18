"""
Example: Running PRESTOS Modules on Different Platforms

This example demonstrates how to use the platform infrastructure to run
transport models on local and remote systems.
"""

import json
from pathlib import Path
import sys

# Ensure PRESTOS is in path
prestos_root = Path(__file__).parent
sys.path.insert(0, str(prestos_root / "src"))

from src.transport import FingerprintsModel
from src.state import PlasmaState
from src.interfaces import gacode
from src.tools.io import PlatformManager, PlatformSpec


def example_1_local_execution():
    """Example 1: Direct local execution (no platform)."""
    print("\n" + "="*70)
    print("Example 1: Direct Local Execution")
    print("="*70)
    
    # Load example plasma state
    gacode_file = prestos_root / "example" / "input.gacode"
    gc = gacode(filepath=str(gacode_file))
    state = PlasmaState.from_gacode(gc)
    state.process(gc)
    
    # Create transport model
    transport = FingerprintsModel(
        options={
            'roa_eval': [0.3, 0.5, 0.7],
            'modes': 'all',
        }
    )
    
    # Evaluate directly
    print(f"Evaluating transport model directly...")
    output_dict, std_dict = transport._evaluate_single(state)
    
    print(f"Output variables: {list(output_dict.keys())}")
    for key, val in output_dict.items():
        print(f"  {key}: {val}")


def example_2_local_platform_execution():
    """Example 2: Local execution through PlatformManager."""
    print("\n" + "="*70)
    print("Example 2: Local Platform Execution")
    print("="*70)
    
    # Load plasma state
    gacode_file = prestos_root / "example" / "input.gacode"
    gc = gacode(filepath=str(gacode_file))
    state = PlasmaState.from_gacode(gc)
    state.process(gc)
    
    # Create transport model
    transport = FingerprintsModel(
        options={
            'roa_eval': [0.3, 0.5, 0.7],
            'modes': 'all',
        }
    )
    
    # Configure local platform
    platform_spec = PlatformSpec(
        name="local_machine",
        machine="local",
        scratch=str(prestos_root / "scratch" / "transport_work"),
        n_cpu=4,
    )
    
    # Evaluate through platform
    print(f"Evaluating transport model on local platform...")
    work_dir = prestos_root / "scratch" / "transport_work"
    output_dict, std_dict = transport.run_on_platform(
        state,
        platform_spec,
        work_dir=work_dir,
        model_name="example_local_run",
        cleanup=False,  # Keep files for inspection
    )
    
    print(f"Output variables: {list(output_dict.keys())}")
    print(f"Work directory: {work_dir}")


def example_3_remote_platform_config():
    """Example 3: Configuration for remote platform (requires setup)."""
    print("\n" + "="*70)
    print("Example 3: Remote Platform Configuration")
    print("="*70)
    
    # Create example platform configuration
    remote_config = {
        "machine": "remote.example.com",
        "username": "user",
        "scratch": "/work/user/prestos",
        "n_cpu": 16,
        "n_gpu": 0,
        "modules": "module load python/3.9",
        "ssh_identity": "~/.ssh/remote_key",
        "scheduler": "none",
    }
    
    print("Example remote platform configuration:")
    print(json.dumps(remote_config, indent=2))
    
    print("\nTo use this platform:")
    print("  1. Update hostname, username, and SSH key path")
    print("  2. Create platform_spec = PlatformSpec.from_dict(remote_config)")
    print("  3. Run: transport.run_on_platform(state, platform_spec)")


def example_4_slurm_platform_config():
    """Example 4: Configuration for SLURM cluster."""
    print("\n" + "="*70)
    print("Example 4: SLURM Cluster Configuration")
    print("="*70)
    
    slurm_config = {
        "machine": "hpc.cluster.org",
        "username": "user",
        "scratch": "/home/user/work",
        "n_cpu": 64,
        "n_gpu": 4,
        "n_ram_gb": 256.0,
        "modules": "module load gcc/11.2.0 && module load openmpi/4.1.0",
        "ssh_identity": "~/.ssh/cluster_key",
        "scheduler": "slurm",
        "slurm_partition": "gpu",
    }
    
    print("Example SLURM cluster platform configuration:")
    print(json.dumps(slurm_config, indent=2))
    
    print("\nTo use this platform:")
    print("  1. Update cluster hostname and credentials")
    print("  2. Verify module names available on cluster")
    print("  3. Create platform_spec = PlatformSpec.from_dict(slurm_config)")
    print("  4. Run: transport.run_on_platform(state, platform_spec)")


def example_5_batch_processing():
    """Example 5: Batch processing multiple states."""
    print("\n" + "="*70)
    print("Example 5: Batch Processing")
    print("="*70)
    
    # Load plasma state
    gacode_file = prestos_root / "example" / "input.gacode"
    gc = gacode(filepath=str(gacode_file))
    state = PlasmaState.from_gacode(gc)
    state.process(gc)
    
    # Create transport model
    transport = FingerprintsModel(
        options={
            'roa_eval': [0.3, 0.5, 0.7],
            'modes': 'all',
        }
    )
    
    # Configure platform
    platform_spec = PlatformSpec(
        name="local_machine",
        machine="local",
        scratch=str(prestos_root / "scratch" / "batch_work"),
        n_cpu=4,
    )
    
    # Process multiple runs
    print("Processing 3 identical transport model evaluations...")
    results = []
    for i in range(3):
        work_dir = prestos_root / "scratch" / f"batch_work_{i}"
        output_dict, std_dict = transport.run_on_platform(
            state,
            platform_spec,
            work_dir=work_dir,
            model_name=f"batch_run_{i}",
            cleanup=True,
        )
        results.append((output_dict, std_dict))
        print(f"  Completed run {i+1}/3")
    
    print(f"\nBatch processing completed. Results collected from 3 runs.")


def example_6_manual_platform_operations():
    """Example 6: Direct PlatformManager operations for custom workflows."""
    print("\n" + "="*70)
    print("Example 6: Direct Platform Manager Operations")
    print("="*70)
    
    # Configure platform
    platform_spec = PlatformSpec(
        name="local_machine",
        machine="local",
        scratch=str(prestos_root / "scratch" / "manual_ops"),
    )
    
    manager = PlatformManager(platform_spec)
    
    try:
        # Create working directory
        work_dir = Path(prestos_root / "scratch" / "manual_ops")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage input files
        print("1. Staging input files...")
        input_dir = prestos_root / "example"
        remote_input_dir = work_dir / "inputs"
        if input_dir.exists():
            manager.stage_inputs(
                input_dir,
                remote_input_dir,
                file_patterns=["input.*"],
            )
            print(f"   Staged files to {remote_input_dir}")
        
        # Execute command
        print("2. Executing command...")
        code, stdout, stderr = manager.run_command(
            "ls -la inputs/",
            cwd=work_dir,
            check=False,
        )
        print(f"   Output: {stdout[:200]}")
        
        # Retrieve outputs
        print("3. Retrieving outputs...")
        output_dir = prestos_root / "scratch" / "manual_outputs"
        manager.retrieve_outputs(remote_input_dir, output_dir)
        print(f"   Retrieved files to {output_dir}")
        
    finally:
        manager.cleanup()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PRESTOS Platform Execution Examples")
    print("="*70)
    
    examples = [
        ("1", "Direct local execution", example_1_local_execution),
        ("2", "Local platform execution", example_2_local_platform_execution),
        ("3", "Remote platform config (reference)", example_3_remote_platform_config),
        ("4", "SLURM platform config (reference)", example_4_slurm_platform_config),
        ("5", "Batch processing", example_5_batch_processing),
        ("6", "Direct platform operations", example_6_manual_platform_operations),
    ]
    
    print("\nAvailable examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    
    print("\nRunning all examples...\n")
    
    for num, desc, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in example {num}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nFor more information, see PLATFORM_SUBMISSION_GUIDE.md")


if __name__ == "__main__":
    main()
