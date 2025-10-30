import optuna
from optuna.samplers import TPESampler
import subprocess
import yaml
import os
from pathlib import Path
import re


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    Suggests hyperparameters and runs training.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu", "tanh"])
    
    # Suggest hidden layer architecture
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_sizes = []
    for i in range(n_layers):
        size = trial.suggest_categorical(f"hidden_size_{i}", [64, 128, 256, 512])
        hidden_sizes.append(size)
    
    # Build command to run training with overridden parameters
    hidden_sizes_str = str(hidden_sizes).replace(" ", "")
    cmd = [
        "python", "train.py", "model=mlp",
        f"model.learning_rate={learning_rate}",
        f"model.dropout={dropout}",
        f"model.activation={activation}",
        f"model.hidden_sizes={hidden_sizes_str}",
        f"hydra.run.dir=outputs/optuna/mlp/trial_{trial.number}",
        f"hydra.job.name=optuna_trial_{trial.number}",
    ]
    
    print(f"\n{'='*80}")
    print(f"Trial {trial.number}")
    print(f"{'='*80}")
    print(f"Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent
        )
        
        # Parse the output to extract val/acc
        output = result.stdout
        
        # Look for validation accuracy in the output
        pattern = r"val[/_]acc[:\s=]+([0-9.]+)"
        matches = re.findall(pattern, output, re.IGNORECASE)
        
        if matches:
            # Get the best (maximum) metric
            metric_value = max(float(m) for m in matches)
            print(f"\nTrial {trial.number} completed with val/acc: {metric_value:.4f}\n")
            return metric_value
        else:
            print(f"\nWarning: Could not extract val/acc from output for trial {trial.number}")
            return 0.0  # Return 0 if metric not found
            
    except subprocess.CalledProcessError as e:
        print(f"\nTrial {trial.number} failed with error:")
        print(e.stderr)
        raise optuna.TrialPruned()


def main():
    """
    Main function to run Optuna optimization.
    """
    # Create study with TPE sampler
    sampler = TPESampler(
        seed=42,
        multivariate=True,
        n_startup_trials=5,  # Number of random trials before TPE kicks in
    )
    
    study = optuna.create_study(
        study_name="mnist_optimization",
        direction="maximize",  # Maximize val/acc
        sampler=sampler,
        storage="sqlite:///optuna_study.db",  # Persist study to database
        load_if_exists=True,
    )
    
    print("="*80)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*80)
    print(f"Study name: {study.study_name}")
    print(f"Sampler: {type(sampler).__name__}")
    print(f"Direction: maximize val/acc")
    print("="*80)
    
    # Run optimization
    n_trials = 50  # Number of trials to run
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1,  # Run trials sequentially
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    # Print results
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (val/acc): {best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters to YAML
    best_params_file = "outputs/best_params.yaml"
    os.makedirs("outputs", exist_ok=True)
    
    with open(best_params_file, "w") as f:
        yaml.dump(best_trial.params, f, default_flow_style=False)
    
    print(f"\nBest parameters saved to: {best_params_file}")
    
    # Print importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter importances:")
        for key, value in importance.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"\nCould not calculate parameter importance: {e}")
    
    # Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        
        # Optimization history
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig("outputs/optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("\nOptimization history plot saved to: outputs/optimization_history.png")
        
        # Parameter importances
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig("outputs/param_importances.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Parameter importances plot saved to: outputs/param_importances.png")
        
        # Parallel coordinate plot
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig.savefig("outputs/parallel_coordinate.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Parallel coordinate plot saved to: outputs/parallel_coordinate.png")
        
    except ImportError:
        print("\nInstall matplotlib to generate visualization plots: pip install matplotlib")
    except Exception as e:
        print(f"\nCould not generate some plots: {e}")


if __name__ == "__main__":
    main()
