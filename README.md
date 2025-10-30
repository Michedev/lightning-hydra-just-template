# lightning-hydra-just-template

A minimal template for AI/ML projects combining PyTorch Lightning, Hydra configuration management, and Just as a task runner.

## Features

- 🔥 **PyTorch Lightning**: Streamlined training loop and model organization
- ⚙️ **Hydra**: Flexible configuration management with composition
- 🚀 **Just**: Simple command runner for common tasks
- 📦 **Conda**: Environment management with Python 3.13

## Project Structure

```
.
├── config/                 # Hydra configuration files
│   ├── dataset/           # Dataset configurations
│   ├── model/             # Model configurations
│   └── train.yaml         # Main training config
├── src/         # Source code
│   ├── model/             # Model implementations
│   └── dataset/           # Dataset implementations
├── train.py               # Training script
├── Justfile               # Task runner commands
└── requirements.txt       # Python dependencies
```

## Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed
- [Just](https://github.com/casey/just) command runner installed

Install Just:
```bash
# macOS
brew install just

# Linux
cargo install just
# or download from releases
```

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd lightning-hydra-just-template
```

2. Create and setup the environment:
```bash
just setup
```

This will:
- Create a conda environment named `env` with Python 3.13
- Install all dependencies from `requirements.txt`

### Available Commands

View all available commands:
```bash
just --list
```

#### Main Commands

- **`just setup`**: Create conda environment and install dependencies
  ```bash
  just setup
  ```

- **`just train`**: Train the MLP model on MNIST
  ```bash
  just train
  ```

- **`just python <script>`**: Run any Python script in the conda environment
  ```bash
  just python train.py model=mlp
  just python -c "import torch; print(torch.__version__)"
  ```
  Alias: `just p <script>`

  ```bash
  just p train.py model=mlp
  just p -c "import torch; print(torch.__version__)"
  ```

- **`just remove-env`**: Remove the conda environment
  ```bash
  just remove-env
  ```

- **`just optimize [TRIALS]`**: Run Optuna hyperparameter optimization
  ```bash
  just optimize          # Run with default 50 trials
  just optimize 100      # Run 100 trials
  ```

- **`just optimize-custom <args>`**: Run optimization with custom arguments
  ```bash
  just optimize-custom --n-trials=100 --seed=123
  just optimize-custom --n-trials=20 --timeout=3600
  ```

### Training with Hydra

The template uses Hydra for configuration. You can override any configuration parameter from the command line:

```bash
# Train with custom learning rate
just python train.py model.learning_rate=0.01

# Train with different batch size
just python train.py batch_size=64

# Train with custom model architecture
just python train.py model.hidden_sizes=[512,256,128]

# Combine multiple overrides
just python train.py model.learning_rate=0.001 model.dropout=0.3 batch_size=128
```

### Customization

#### Modify Environment Settings

Edit the `Justfile` to change environment name or Python version:

```just
env_name := "my_env"        # Change environment name
python_version := "3.11"    # Change Python version
```

#### Add New Models

1. Create a new model file in `your_codebase/model/`
2. Add corresponding config in `config/model/`
3. Train with: `just train model=<your_model>`

#### Add New Datasets

1. Create a new dataset file in `your_codebase/dataset/`
2. Add corresponding config in `config/dataset/`
3. Update training config to use your dataset

## Example: MNIST MLP Training

The template includes a complete example with:
- MLP model with configurable architecture
- Flattened MNIST dataset
- Training script with validation

Run the example:
```bash
just setup
just train
```

## Hyperparameter Optimization with Optuna

The template includes Optuna for automated hyperparameter tuning. The optimization script systematically searches for the best hyperparameters.

#### Quick Start

Run optimization with default settings (50 trials):
```bash
just optimize
```

Run with custom number of trials:
```bash
just optimize 100
```

#### Advanced Usage

The optimization script supports various command-line arguments:

```bash
# Run 200 trials with custom study name
just optimize-custom --n-trials=200 --study-name=my_experiment

# Continue existing study with more trials
just optimize-custom --n-trials=50 --study-name=my_experiment

# Set timeout (in seconds)
just optimize-custom --n-trials=100 --timeout=7200

# Use custom database storage
just optimize-custom --n-trials=50 --storage=postgresql://user:pass@host/db

# Set random seed for reproducibility
just optimize-custom --n-trials=50 --seed=42

# Customize TPE sampler startup trials
just optimize-custom --n-trials=100 --n-startup-trials=10
```

#### Available Arguments

- `--n-trials`: Number of optimization trials to run (default: 50)
- `--study-name`: Name of the Optuna study (default: "mlp_mnist_optimization")
- `--storage`: Database URL for storing study results (default: "sqlite:///optuna_mlp.db")
- `--n-startup-trials`: Number of random trials before TPE optimization (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--timeout`: Maximum time in seconds for optimization (default: None)

#### Hyperparameters Being Optimized

The template optimizes the following hyperparameters:
- **Learning rate**: 1e-5 to 1e-2 (log scale)
- **Dropout**: 0.0 to 0.5
- **Activation function**: relu, gelu, leaky_relu, tanh
- **Number of layers**: 1 to 4
- **Hidden layer sizes**: 64, 128, 256, or 512 neurons per layer

#### Output

Results are saved to:
- `outputs/best_mlp_params.yaml`: Best hyperparameters in YAML format
- `outputs/optimization_history.png`: Plot of optimization progress
- `outputs/param_importances.png`: Parameter importance visualization
- `outputs/parallel_coordinate.png`: Parallel coordinate plot
- `optuna_mlp.db`: SQLite database with all trial results

#### Resume Interrupted Studies

Studies are automatically saved to the database. You can resume interrupted optimizations:

```bash
# First run
just optimize 100

# If interrupted, continue with same study name
just optimize-custom --n-trials=50 --study-name=mlp_mnist_optimization
```

#### Customizing the Optimization

To customize what hyperparameters are optimized:

1. Edit `optuna_mlp_validation.py` or copy to `optuna_validation.py`
2. Modify the `objective()` function to suggest different hyperparameters
3. Update the command-line call to match your model configuration
4. Run your custom optimization script

Example for custom model:
```python
# In objective function
learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

cmd = [
    "python", "train.py", "model=my_model",
    f"model.learning_rate={learning_rate}",
    f"batch_size={batch_size}",
    # ...
]
```
