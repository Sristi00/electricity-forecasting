# OVERWRITE EXACTLY: src/gnn_train.py
"""
Training Module - Train Spatial-Temporal Homogeneous GNN models for electricity price forecasting
"""
import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error, r2_score

from gnn_models import create_model
from gnn_config import GNN_CONFIG, DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from gnn_graph_builder import load_graph


class GNNTrainer:
    """Trainer for GNN models operating over flat spatial-temporal grids."""
    
    def __init__(self, model, data, config=None):
        """
        Initialize trainer.
        
        Args:
            model: GNN model instance.
            data: PyTorch Geometric Data object.
            config: Training configuration dictionary.
        """
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.config = config or GNN_CONFIG
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch across all active node coordinates."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss on training nodes (includes historical states of all 4 zones)
        loss = F.mse_loss(out[self.data.train_mask].view(-1), self.data.y[self.data.train_mask].view(-1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute MAE
        with torch.no_grad():
            mae = F.l1_loss(out[self.data.train_mask].view(-1), self.data.y[self.data.train_mask].view(-1))
        
        return loss.item(), mae.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate model performance on the target verification mask."""
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        
        loss = F.mse_loss(out[mask].view(-1), self.data.y[mask].view(-1))
        mae = F.l1_loss(out[mask].view(-1), self.data.y[mask].view(-1))
        
        return loss.item(), mae.item()
    
    def train(self, num_epochs=None, early_stopping_patience=None):
        """Train the homogeneous spatial-temporal model."""
        num_epochs = num_epochs or self.config['num_epochs']
        early_stopping_patience = early_stopping_patience or self.config['early_stopping_patience']
        
        print(f"\n🚀 Starting training loop for {num_epochs} epochs...")
        print(f"Device: {DEVICE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_mae = self.train_epoch()
            val_loss, val_mae = self.evaluate(self.data.val_mask)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train MAE: {train_mae:.4f} | "
                      f"Val MAE: {val_mae:.4f}")
            
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break
        
        print(f"\n✅ Training sequence complete!")
        return self.history
    
    def test(self):
        print("\n📊 Evaluating on test set...")
        
        # Reload structural parameters
        self.load_checkpoint('best_model.pt')
        
        # Load target scaler to handle inverse feature translations
        scaler_path = GRAPH_DIR / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            target_scaler = scalers['target_scaler']
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            
            # Extract target indices corresponding to the test block array
            y_pred_scaled = out[self.data.test_mask].cpu().numpy().reshape(-1, 1)
            y_true_scaled = self.data.y[self.data.test_mask].cpu().numpy().reshape(-1, 1)
            
            # Invert back to real-scale DKK market values
            y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
            y_true = target_scaler.inverse_transform(y_true_scaled).flatten()
            
            # Structural assessment calculation
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            r2 = float(r2_score(y_true, y_pred))
            
            # Robust SMAPE fallback to bypass zero/negative market values cleanly
            smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': smape
        }
        
        print(f"\n📈 Real-Scale Multi-Zone Test Results (DK1 + DK2):")
        print(f"   MAE:  {mae:.4f} DKK")
        print(f"   RMSE: {rmse:.4f} DKK")
        print(f"   R²:   {r2:.4f}")
        print(f"   SMAPE: {smape:.2f}%")
        
        return metrics
    
    def save_checkpoint(self, filename):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ARTIFACTS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
        checkpoint_path = ARTIFACTS_DIR / filename
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
    
    def save_history(self, filename='training_history.json'):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        history_path = ARTIFACTS_DIR / filename
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 Saved training history data to {history_path}")


def train_gnn_model(data, model_type='GCN', config=None):
    config = config or GNN_CONFIG
    num_features = data.x.shape[1]
    
    model = create_model(model_type, num_features, config)
    
    print(f"\n🔧 Initializing {model_type} Core Model Architecture...")
    print(f"   Input features  : {num_features}")
    print(f"   Hidden channels : {config['hidden_channels']}")
    print(f"   Network layers  : {config['num_layers']}")
    
    trainer = GNNTrainer(model, data, config)
    history = trainer.train()
    metrics = trainer.test()
    
    # Save the evaluation dictionary package into history format for compare_models.py
    trainer.history['val_mae'] = metrics['mae']
    trainer.history['val_rmse'] = metrics['rmse']
    trainer.history['val_r2'] = metrics['r2']
    trainer.save_history()
    
    return model, metrics, history


if __name__ == "__main__":
    # 1. Fetch graph from builder file
    data = load_graph()
    
    # 2. Fire optimization execution run
    model, metrics, history = train_gnn_model(data, model_type='GCN')
    print("\n✅ Homogeneous Spatial-Temporal Mesh Baseline Run Finalized!")