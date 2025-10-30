env_name := "env"
python_version := "3.13"

alias p := python

setup:
    conda create -n {{env_name}} python={{python_version}} -y
    conda run -n {{env_name}} pip install -r requirements.txt

# Alternative setup using venv instead of conda
# Uncomment and use this if you prefer venv over conda
# setup-venv:
#     python{{python_version}} -m venv {{env_name}}
#     {{env_name}}/bin/pip install --upgrade pip
#     {{env_name}}/bin/pip install -r requirements.txt

remove-env:
    conda env remove -n {{env_name}}

# Remove venv environment (use with setup-venv)
# remove-venv:
#     rm -rf {{env_name}}

train:
    conda run -n {{env_name}} python train.py model=mlp

# Run Optuna hyperparameter optimization
optimize TRIALS="50":
    conda run -n {{env_name}} python optuna_mlp_validation.py --n-trials={{TRIALS}}

# Continue an existing Optuna study with additional trials
optimize-continue TRIALS="50":
    conda run -n {{env_name}} python optuna_mlp_validation.py --n-trials={{TRIALS}}

# Run optimization with custom parameters
optimize-custom *ARGS:
    conda run -n {{env_name}} python optuna_mlp_validation.py {{ARGS}}

python *ARGS:
    conda run -n {{env_name}} --live-stream python {{ARGS}}

# Run python with venv (use with setup-venv)
# python-venv *ARGS:
#     {{env_name}}/bin/python {{ARGS}}