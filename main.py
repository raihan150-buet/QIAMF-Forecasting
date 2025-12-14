import argparse
import torch
import wandb
from configs.config import CONFIG
from src.data.dataset import prepare_data
from src.models.qiamf import QIAMF
from src.training.loss import NovelLoss
from src.training.trainer import Trainer

def main(args):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing QIAMF using {device}...")
    
    # 2. WandB Init
    if not args.no_wandb:
        wandb.init(project=CONFIG['project_name'], name=CONFIG['experiment_name'], config=CONFIG)
    else:
        # Mock wandb if disabled
        wandb.init(mode="disabled")

    # 3. Data
    print("Preparing Data...")
    train_loader, val_loader, scaler, input_dim = prepare_data(
        args.data_path, 
        CONFIG['seq_length'], 
        CONFIG['output_dim'],
        CONFIG['train_ratio'],
        CONFIG['batch_size']
    )
    
    # 4. Model & Optimizer
    model = QIAMF(input_dim, CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = NovelLoss(CONFIG)
    
    # 5. Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, CONFIG, device)
    
    # 6. Resume or Inference
    if args.resume:
        trainer.load_checkpoint(f"{CONFIG['save_dir']}/last.pth")
    elif args.load_best:
        trainer.load_checkpoint(f"{CONFIG['save_dir']}/best.pth")
        
    if not args.inference_only:
        trainer.train()
    else:
        print("Running Inference Evaluation...")
        loss, mse = trainer.evaluate()
        print(f"Inference Results - Loss: {loss:.4f}, MSE: {mse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QIAMF Training Script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--load_best', action='store_true', help='Load best model weights')
    parser.add_argument('--inference_only', action='store_true', help='Skip training, just evaluate')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    main(args)