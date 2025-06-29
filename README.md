Pulsar Timing Analysis with TEMPO2 and MCMC

This repository contains a Python script (TempoLike9.py) designed to perform pulsar timing analysis by interfacing with TEMPO2 and employing Markov Chain Monte Carlo (MCMC) methods for parameter estimation. It allows for the exploration of parameter space, including fitting for timing model parameters, JUMP parameters, and T2EFACs, while incorporating both .par file priors and optional custom theory priors.
Features

    Parameter Prior Extraction: Automatically extracts priors for fitting parameters (e.g., RA, Dec, JUMPs, T2EFACs) directly from a TEMPO2 .par file.

    Custom Theory Priors: Supports an optional external file (theory_priors.txt) to define additional uniform priors and physical equations for derived parameters, with these custom priors taking precedence.

    Derived Parameter Computation: Dynamically computes derived parameters based on user-defined equations using sympy, ensuring physical consistency and handling potential invalid computations (e.g., division by zero).

    TEMPO2 Integration: Seamlessly runs TEMPO2 with updated .par files for each MCMC step to generate timing residuals.

    Likelihood Calculation: Computes the likelihood of a given set of parameters based on the TEMPO2 residuals and their uncertainties, correctly applying T2EFAC corrections.

    MCMC Sampling (emcee): Utilizes the emcee library to perform MCMC sampling, allowing for robust exploration of the parameter posterior distribution.

    Corner Plots: Generates corner plots using corner.py to visualize the 1D and 2D marginalized posterior distributions of sampled and derived parameters.

    Error Handling and Debugging: Includes robust error handling for TEMPO2 failures (saving problematic .par files), checks for invalid uncertainties, and provides verbose warnings.

Getting Started
Prerequisites

    Python 3.x

    numpy

    emcee

    corner

    matplotlib

    sympy

    TEMPO2 (and its environment variables correctly set up)

You can install the Python dependencies using pip:

pip install numpy emcee corner matplotlib sympy

Installation

Clone the repository:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

Usage

    Prepare your TEMPO2 files:

        Place your pulsar .par file and .tim file in the same directory as the script, or provide their paths.

    Optional: Create a theory_priors.txt file:
    This file can define additional uniform priors for parameters and equations for derived parameters. Each line should be either a prior definition or an equation.

        Prior format: PARAMETER_NAME LOWER_BOUND UPPER_BOUND
        Example: PB 0.07 0.08

        Equation format: DERIVED_PARAMETER = EQUATION_EXPRESSION
        Example: M_comp = (PB * (M_pulsar * sin(I))**2) / (2 * PI * G_Newton / cFactor)**(1/3)
        (Note: Ensure sympy compatible syntax for equations. M_pulsar and I would need to be sampled parameters or have fixed values.)

    Run the script:

    python TempoLike9.py <PAR_FILE> <TIM_FILE> [--theory_priors THEORY_PRIORS_FILE] [--delta DELTA_FACTOR] [--nwalkers NWALKERS] [--nsteps NSTEPS] [--burnin BURNIN] [--threads THREADS] [--output_prefix OUTPUT_PREFIX]

    Arguments:

        <PAR_FILE>: Path to the TEMPO2 pulsar parameter file.

        <TIM_FILE>: Path to the TEMPO2 pulsar timing data file.

        --theory_priors THEORY_PRIORS_FILE: (Optional) Path to the file containing additional uniform priors and equations for derived parameters.

        --delta DELTA_FACTOR: (Optional) Factor to multiply the uncertainties from the .par file to define uniform prior ranges (default: 10.0).

        --nwalkers NWALKERS: (Optional) Number of walkers for the MCMC sampler (default: 50).

        --nsteps NSTEPS: (Optional) Number of steps for each MCMC chain (default: 1000).

        --burnin BURNIN: (Optional) Number of burn-in steps to discard (default: 200).

        --threads THREADS: (Optional) Number of CPU threads to use for parallelizing the MCMC (default: 1). Use -1 to use all available CPU cores.

        --output_prefix OUTPUT_PREFIX: (Optional) Prefix for output files (e.g., corner plot, MCMC chain).

    Example:

    python TempoLike9.py J1713+0747.par J1713+0747.tim --theory_priors my_theory_priors.txt --nwalkers 100 --nsteps 2000 --burnin 500 --threads 4 --output_prefix pulsar_analysis

Output

The script will generate:

    A corner plot (.png file) showing the posterior distributions of the sampled and derived parameters.

    Best-fit values (median with 1-sigma uncertainties) for all sampled and derived parameters printed to the console.

    A chain.npy file containing the raw MCMC chain.

    Up to 5 sample_*.par files (for debugging purposes) that were successfully generated during the run.

    A failed_pars directory if TEMPO2 fails for any MCMC sample, containing the .par file that caused the failure for debugging.

Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.


