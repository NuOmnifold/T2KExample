# OmnifoldT2K

A framework for applying the OmniFold unfolding procedure to T2K neutrino experiment data.

## Overview

This repository implements the OmniFold algorithm to unfold detector-level distributions back to truth-level (generator) distributions for T2K experiment data. The unfolding process corrects for detector effects, selection biases, and other experimental limitations to recover the underlying physics distributions.

## Key Components

- **t2k.py**: Main executable script that sets up the unfolding procedure, loads configurations, and orchestrates the workflow
- **omnifold.py**: Core implementation of the OmniFold unfolding procedure based on iterative neural network reweighting
- **utils.py**: Utility functions for data handling, visualization, and analysis
- **GetOmnifoldWeights.py**: Script to extract and save the final reweighting factors after the unfolding procedure
- **config_omnifold.json**: Configuration file with model parameters and file paths
- **SingleIterationDemo.ipynb**: Jupyter notebook demonstrating a single iteration of the OmniFold procedure with detailed explanations and visualizations

## How It Works

The OmniFold procedure uses a two-step iterative approach:

1. **Step 1 (Reweighting Detection Level)**: Train a model to distinguish between real data and simulated reconstructed data, then use this model to reweight the simulated data
2. **Step 2 (Reweighting Generator Level)**: Train a model to distinguish between the original simulated generator-level events and the same events reweighted by Step 1, then apply this model to unfold back to truth-level

This process is repeated for multiple iterations to improve the unfolding accuracy.

## Key Features

- **Distributed Training**: Uses Horovod to enable multi-GPU training
- **Visualization**: Creates diagnostic plots showing the reweighting at each step
- **Customizable Configuration**: Easy parameter adjustment through JSON configuration file
- **Ensemble Training**: Supports multiple trials to improve robustness

## Data Structure

The framework expects data in NumPy array format:
- Detector-level (reco) data from real experiment
- Detector-level (reco) simulated data
- Generator-level (gen/truth) simulated data
- Associated weights and selection masks

## Usage

### Basic Usage

```bash
./runOmnifold.sh
```

This script will run the full unfolding procedure using the default configuration in `config_omnifold.json`. The script:

```bash
#!/bin/bash

module load tensorflow/2.9.0
python t2k.py --config config_omnifold.json --weights_folder weights_omnifold/ \
  --file_path /global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/FormattedData_v13/ \
  --no_eff --verbose
```

It loads the TensorFlow module and executes the main script with default configuration, specifying weights folder and data paths, while enabling verbose output and the "no_eff" option (which omits truth events not reconstructed).

### Advanced Options

```bash
python t2k.py --config config_omnifold.json --plot_folder ./plots/ \
  --weights_folder ./weights/ --file_path /path/to/data/ \
  --nevts -1 --verbose --start_iter 0
```

Parameters:
- `--config`: Path to configuration file
- `--plot_folder`: Directory to save diagnostic plots
- `--weights_folder`: Directory to save model weights
- `--file_path`: Path to input data files
- `--nevts`: Number of events to use (-1 for all)
- `--verbose`: Enable detailed logging
- `--shape_only`: Normalize distributions to focus on shape differences
- `--start_iter`: Starting iteration number (for resuming)
- `--no_eff`: Omit truth events not reconstructed in step 2
- And more...

### Extracting Final Weights

```bash
python GetOmnifoldWeights.py --start_iter 0 --total_iter 15 \
  --trials 3 --throws 100 --save_name "MyAnalysis" \
  --weight_prefix "t2k_FDS0_MCStat" --weight_dir "./weights/"
```

This script extracts the final reweighting factors after the unfolding procedure.

## Configuration

The `config_omnifold.json` file contains:

```json
{
  "FILE_MC_RECO": "mc_vals_reco_Nominal.npy",
  "FILE_MC_GEN": "mc_vals_truth_Topology.npy",
  "FILE_MC_FLAG_RECO": "mc_pass_reco_Nominal.npy",
  "FILE_MC_FLAG_GEN": "mc_pass_truth_Nominal.npy",
  "FILE_DATA_RECO": "mc_vals_reco_Nominal.npy",
  "FILE_DATA_FLAG_RECO": "mc_pass_reco_Nominal.npy",
  "FILE_DATA_WEIGHT": "mc_weights_reco_FakeDataStudy6.npy",
  "FILE_MC_RECO_WEIGHT": "mc_weights_reco_Nominal.npy",
  "FILE_MC_GEN_WEIGHT": "mc_weights_truth_Nominal.npy",
  "NITER": 15,
  "NTRIAL": 3,
  "LR": 1e-4,
  "BATCH_SIZE": 1024,
  "EPOCHS": 500,
  "NWARMUP": 5,
  "NAME": "t2k_Nominal",
  "NPATIENCE": 15
}
```

These parameters control file paths, neural network hyperparameters, and training settings.

## Dependencies

- TensorFlow/Keras
- Horovod (for distributed training)
- NumPy
- Matplotlib
- SciPy

## Model Architecture

The neural networks used in the unfolding process are multi-layer perceptrons (MLPs) with:
- Multiple dense layers with leaky ReLU activation
- Sigmoid output layer
- Binary cross-entropy loss with custom weighting

## Example Notebook

The repository includes `SingleIterationDemo.ipynb`, which provides:

- A step-by-step demonstration of a single OmniFold iteration
- Code for loading and preprocessing T2K data
- Implementation of both Step 1 (reco-level reweighting) and Step 2 (truth-level reweighting)
- Visualization of results at each stage of the unfolding process
- Comparison between original, pulled, and unfolded distributions

This notebook is an excellent starting point for understanding how OmniFold works in practice with T2K data.

## Contributing

To contribute, please follow the standard GitHub workflow:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

This implementation is based on the OmniFold methodology described in:
- "OmniFold: A Method to Simultaneously Unfold All Observables" by Andreassen, Komiske, Metodiev, et al.