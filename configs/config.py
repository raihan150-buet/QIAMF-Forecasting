# Central configuration dictionary
CONFIG = {
    # Data params
    "seq_length": 168,      # 7 days lookback
    "output_dim": 24,       # 24 hours prediction
    "train_ratio": 0.8,
    
    # Model params
    "d_model": 128,
    "dropout": 0.1,
    
    # Training params
    "batch_size": 32,
    "learning_rate": 0.0005,
    "epochs": 50,
    "patience": 15,
    "save_dir": "./checkpoints",
    
    # Loss weights
    "lambda_uncertainty": 0.1,
    "lambda_consistency": 0.05,
    "lambda_decomp": 0.01,
    
    # WandB
    "project_name": "qiamf-thesis",
    "experiment_name": "run-v1"
}