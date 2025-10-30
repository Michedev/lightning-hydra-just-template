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

- **`just remove-env`**: Remove the conda environment
  ```bash
  just remove-env
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

## License

MIT
