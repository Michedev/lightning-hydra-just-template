env_name := "env"
python_version := "3.13"

alias p := python

setup:
    conda create -n {{env_name}} python={{python_version}} -y
    conda run -n {{env_name}} pip install -r requirements.txt
remove-env:
    conda env remove -n {{env_name}}
train:
    conda run -n {{env_name}} python train.py model=mlp

# Run Optuna hyperparameter optimization
optimize:
    conda run -n {{env_name}} python optuna_mlp_validation.py

# Continue an existing Optuna study with additional trials
optimize-continue TRIALS="50":
    conda run -n {{env_name}} python optuna_mlp_validation.py --n-trials={{TRIALS}}

python *ARGS:
    conda run -n {{env_name}} --live-stream python {{ARGS}}