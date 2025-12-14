import torch
import os
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.start_epoch = 0
        self.best_loss = float('inf')
        
    def save_checkpoint(self, epoch, is_best=False):
        os.makedirs(self.config['save_dir'], exist_ok=True)
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        # Save latest
        torch.save(state, os.path.join(self.config['save_dir'], 'last.pth'))
        # Save best
        if is_best:
            torch.save(state, os.path.join(self.config['save_dir'], 'best.pth'))
            print(f"  [Checkpoint] Best model saved at epoch {epoch+1}")

    def load_checkpoint(self, path):
        if os.path.exists(path):
            print(f"  [Checkpoint] Loading from {path}...")
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"  [Checkpoint] Resumed from epoch {self.start_epoch}")
        else:
            print("  [Checkpoint] No checkpoint found. Starting from scratch.")

    def train(self):
        print(f"\nStarting training on {self.device}...")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            self.model.train()
            train_loss_sum = 0
            
            # Training Loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(X)
                loss, metrics = self.loss_fn(out, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss_sum += metrics['loss']
                pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
                
                # Wandb Step Log
                wandb.log({"train_step_loss": metrics['loss']})
            
            avg_train_loss = train_loss_sum / len(self.train_loader)
            
            # Validation Loop
            val_loss, val_mse = self.evaluate()
            
            # Logging
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_mse": val_mse
            })
            
            print(f"  Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Checkpointing
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_mse = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                loss, metrics = self.loss_fn(out, y)
                total_loss += metrics['loss']
                total_mse += metrics['mse']
        
        return total_loss / len(self.val_loader), total_mse / len(self.val_loader)