import numpy as np

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

class TGLFHandler:
    """
    High-level orchestration class for managing TGLF simulations within
    reduced-model or transport-optimization workflows.

    This class is responsible for:
      1. Generating TGLF input files from a specified magnetic geometry
         and dimensionless parameter set.
      2. Submitting simulations to the user-configured execution platform
         (local workstation, Perlmutter, or other HPC system).
      3. Monitoring simulation progress and termination status.
      4. Staging and retrieving TGLF output files for downstream analysis.  
    
    Physics post-processing (e.g. flux-surface averaging, transport coefficient
    extraction) is intentionally delegated to separate classes.

    The handler is designed to integrate naturally with automated workflows
    using surrogate modeling, Bayesian optimization, or saturation detection
    tools such as QUENDS.

    """

    def __init__(self, geometry, run_config, platform):
        """
        Initialize a TGLFHandler instance.

        Parameters
        ----------
        geometry : object
            Magnetic geometry specification used by TGLF. This may encapsulate:
              - poloidal flux psi(R,Z),
              - magnetic field components,
              - boundary topology and limiter/divertor geometry.
            The geometry is assumed fixed across parameter scans unless explicitly
            varied.

        run_config : dict
            Dictionary defining numerical and physical configuration options,
            including:
              - model type (e.g. kinetic, gyrofluid, extended-fluid),
              - polynomial order,
              - grid resolution,
              - species definitions,
              - collision models,
              - normalization choices.

        platform : PlatformConfig
            Platform-specific execution configuration, including:
              - execution mode (local, batch),
              - scheduler type (SLURM, none),
              - GPU/CPU resource requests,
              - environment modules or containers.
        """

    # Further methods would be defined here following the same pattern as GkeyllHandler


class GkeyllHandler:
    """
    High-level orchestration class for managing Gkeyll simulations within
    reduced-model or transport-optimization workflows.

    This class is responsible for:
      1. Generating Gkeyll input files (Lua) from a specified magnetic geometry
         and dimensionless parameter set.
      2. Submitting simulations to the user-configured execution platform
         (local workstation, Perlmutter, or other HPC system).
      3. Monitoring simulation progress and termination status.
      4. Staging and retrieving Gkeyll output files for downstream analysis.

    Physics post-processing (e.g. flux-surface averaging, transport coefficient
    extraction) is intentionally delegated to separate classes.

    The handler is designed to integrate naturally with automated workflows
    using surrogate modeling, Bayesian optimization, or saturation detection
    tools such as QUENDS.
    """

    def __init__(self, geometry, run_config, platform):
        """
        Initialize a GkeyllHandler instance.

        Parameters
        ----------
        geometry : object
            Magnetic geometry specification used by Gkeyll. This may encapsulate:
              - poloidal flux psi(R,Z),
              - magnetic field components,
              - boundary topology and limiter/divertor geometry.
            The geometry is assumed fixed across parameter scans unless explicitly
            varied.

        run_config : dict
            Dictionary defining numerical and physical configuration options,
            including:
              - model type (e.g. kinetic, gyrofluid, extended-fluid),
              - polynomial order,
              - grid resolution,
              - species definitions,
              - collision models,
              - normalization choices.

        platform : PlatformConfig
            Platform-specific execution configuration, including:
              - execution mode (local, batch),
              - scheduler type (SLURM, none),
              - GPU/CPU resource requests,
              - environment modules or containers.
        """

    def generate_input(self, parameters, output_dir):
        """
        Generate a Gkeyll Lua input file for a specified parameter set.

        Parameters
        ----------
        parameters : dict
            Dictionary of dimensionless and dimensional parameters defining
            the simulation, e.g.:
              - a/L_ne, a/L_Te, a/L_Ti,
              - beta, rho_star, nu_star,
              - reference density and temperature.
            These parameters are mapped onto background profiles and source terms
            in the Lua input.

        output_dir : str
            Directory in which the input file and associated run artifacts
            will be created.

        Returns
        -------
        input_path : str
            Path to the generated Gkeyll input file.
        """

    def submit(self, input_path):
        """
        Submit a Gkeyll simulation for execution.

        Depending on the platform configuration, this may:
          - launch a local Gkeyll process,
          - generate and submit a batch job script (e.g. SLURM),
          - request GPU resources as specified.

        Parameters
        ----------
        input_path : str
            Path to the Gkeyll input file to be executed.

        Returns
        -------
        run_id : str
            Identifier for the submitted run (job ID or local process ID).
        """

    def monitor(self, run_id, check_interval=60):
        """
        Monitor the execution status of a running Gkeyll simulation.

        This method periodically checks for:
          - job completion or failure,
          - runtime errors,
          - presence of expected output files.

        It is designed to be compatible with external saturation-detection
        tools (e.g. QUENDS), which may independently analyze partial outputs
        to determine when statistically steady behavior has been reached.

        Parameters
        ----------
        run_id : str
            Identifier returned by the submit() method.

        check_interval : int, optional
            Time in seconds between successive status checks.
        """

    def retrieve_outputs(self, run_id, destination):
        """
        Retrieve and stage Gkeyll output files after run completion.

        Parameters
        ----------
        run_id : str
            Identifier of the completed simulation.

        destination : str
            Local directory to which output files (e.g. field data, moments,
            diagnostics) will be copied.

        Returns
        -------
        output_manifest : dict
            Dictionary listing retrieved files and their locations, suitable
            for downstream processing by flux-extraction or diagnostic tools.
        """

    def run(self, parameters, workdir):
        """
        Convenience method to execute a complete Gkeyll workflow:
          - input generation,
          - submission,
          - monitoring,
          - output retrieval.

        This method does not perform physics analysis or time averaging; it
        only guarantees that the raw simulation data are available for
        post-processing.

        Parameters
        ----------
        parameters : dict
            Parameter set defining the simulation.

        workdir : str
            Working directory for the run.

        Returns
        -------
        output_manifest : dict
            Dictionary describing the retrieved output files.
        """
