import numpy as np
import os
import joblib
from typing import Dict, Any

from src.config import DEVICE, ALL_FEATURES, SEQUENCE_LENGTH, LSTM_PARAMS

# Global flag to track if PyTorch can be successfully loaded and initialized
USE_PYTORCH_AE = True

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _t = torch.tensor([1.0])
except (ImportError, OSError) as e:
    print(f"\n========================================================")
    print(f" [SENTRY-AI WARNING] PyTorch DLL loading failed on this system:")
    print(f" {e}")
    print(f" Switching ML Core to high-resilience fallback:")
    print(f" scikit-learn MLP Sequence Autoencoder (No DLL dependencies)")
    print(f"========================================================\n")
    USE_PYTORCH_AE = False


# ----------------------------------------------------
# 1. PyTorch LSTM Module Definition
# ----------------------------------------------------
if USE_PYTORCH_AE:
    class PyTorchLSTMAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, seq_len: int):
            super(PyTorchLSTMAutoencoder, self).__init__()
            self.seq_len = seq_len
            self.input_dim = input_dim
            
            # Encoder
            self.encoder_lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0
            )
            self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
            
            # Decoder
            self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
            self.decoder_lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0
            )
            self.decoder_output = nn.Linear(hidden_dim, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            enc_out, _ = self.encoder_lstm(x)
            last_hidden = enc_out[:, -1, :] 
            latent = torch.relu(self.encoder_fc(last_hidden))
            
            dec_input = self.decoder_fc(latent)
            dec_input_repeated = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1)
            
            dec_out, _ = self.decoder_lstm(dec_input_repeated)
            reconstruction = self.decoder_output(dec_out)
            return reconstruction
else:
    class PyTorchLSTMAutoencoder:
        pass


# ----------------------------------------------------
# 2. Main Wrapper class with self-healing fallback & Soft Bounding
# ----------------------------------------------------
class AnomalyLSTMAutoencoder:
    def __init__(self, feature_dim: int = len(ALL_FEATURES), params: Dict[str, Any] = LSTM_PARAMS):
        self.feature_dim = feature_dim
        self.seq_len = SEQUENCE_LENGTH
        self.hidden_dim = params["hidden_dim"]
        self.latent_dim = params["latent_dim"]
        self.num_layers = params["num_layers"]
        
        self.threshold = 0.5
        self.mu = 0.0
        self.sigma = 1.0
        self.is_fitted = False
        
        if USE_PYTORCH_AE:
            self.lr = params["lr"]
            self.batch_size = params["batch_size"]
            self.epochs = params["epochs"]
            self.model = PyTorchLSTMAutoencoder(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_layers=self.num_layers,
                seq_len=self.seq_len
            ).to(DEVICE)
        else:
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(self.hidden_dim, self.latent_dim, self.hidden_dim),
                activation="relu",
                solver="adam",
                learning_rate_init=0.005,
                max_iter=30,
                batch_size=256,
                random_state=42
            )

    def fit(self, X: np.ndarray):
        """
        Train the Sequence Autoencoder strictly on clean normal transaction sequences.
        X has shape (N, seq_len, feature_dim)
        """
        if USE_PYTORCH_AE:
            print(f"Training PyTorch LSTM Autoencoder on {X.shape[0]} normal sequences...")
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch in dataloader:
                    x_batch = batch[0].to(DEVICE)
                    optimizer.zero_grad()
                    reconstructed = self.model(x_batch)
                    loss = criterion(reconstructed, x_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * x_batch.size(0)
                epoch_loss /= len(dataloader.dataset)
                if (epoch + 1) % 2 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] LSTM MSE Loss: {epoch_loss:.6f}")
                
            # Calibrate normal statistics
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                recon = self.model(x_tensor)
                errors = torch.mean((recon - x_tensor) ** 2, dim=(1, 2)).cpu().numpy()
                
                # Robustly prune top 2% of errors to avoid scale squashing
                q = np.percentile(errors, 98.0)
                clean_errors = errors[errors <= q]
                self.mu = float(np.mean(clean_errors))
                self.sigma = float(np.std(clean_errors))
                if self.sigma == 0.0:
                    self.sigma = 1.0
        else:
            print(f"Training scikit-learn MLP Autoencoder on {X.shape[0]} sequences...")
            N, S, F = X.shape
            X_flat = X.reshape(N, S * F)
            
            self.model.fit(X_flat, X_flat)
            
            # Calculate errors on training set to calibrate range
            X_recon = self.model.predict(X_flat)
            errors = np.mean((X_flat - X_recon) ** 2, axis=1)
            
            # Robustly prune top 2% of errors to avoid scale squashing
            q = np.percentile(errors, 98.0)
            clean_errors = errors[errors <= q]
            self.mu = float(np.mean(clean_errors))
            self.sigma = float(np.std(clean_errors))
            if self.sigma == 0.0:
                self.sigma = 1.0
            
        self.is_fitted = True
        print(f"Sequence Autoencoder trained. Z-Score parameters: mu={self.mu:.4f}, sigma={self.sigma:.4f}")

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate sequence reconstruction-based anomaly probability using Soft Bounding.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet!")
            
        if USE_PYTORCH_AE:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                recon = self.model(x_tensor)
                errors = torch.mean((recon - x_tensor) ** 2, dim=(1, 2)).cpu().numpy()
        else:
            N, S, F = X.shape
            X_flat = X.reshape(N, S * F)
            X_recon = self.model.predict(X_flat)
            errors = np.mean((X_flat - X_recon) ** 2, axis=1)
            
        # Compute Soft Bounding: Sigmoid applied to Z-Score
        z_scores = (errors - self.mu) / self.sigma
        scaled_scores = 1.0 / (1.0 + np.exp(-z_scores))
        
        return scaled_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)

    def calibrate_threshold(self, X_val: np.ndarray, contamination: float = 0.01):
        scores = self.predict_score(X_val)
        self.threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
        print(f"Sequence Autoencoder calibrated Soft threshold: {self.threshold:.4f}")

    def save(self, filepath: str):
        """Save model checkpoints based on active model type."""
        state = {
            "is_pytorch": USE_PYTORCH_AE,
            "threshold": self.threshold,
            "mu": self.mu,
            "sigma": self.sigma,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers
        }
        
        if USE_PYTORCH_AE:
            state["model_state"] = self.model.state_dict()
            torch.save(state, filepath)
            print(f"PyTorch LSTM Autoencoder saved to {filepath}")
        else:
            state["model"] = self.model
            joblib_path = filepath.replace(".pth", "_mlp.joblib")
            joblib.dump(state, joblib_path)
            print(f"scikit-learn MLP Autoencoder saved to {joblib_path}")

    def load(self, filepath: str):
        """Load model checkpoints based on serialization format."""
        joblib_path = filepath.replace(".pth", "_mlp.joblib")
        
        if not USE_PYTORCH_AE or os.path.exists(joblib_path):
            if not os.path.exists(joblib_path):
                raise FileNotFoundError(f"Fallback checkpoint file not found at {joblib_path}")
            state = joblib.load(joblib_path)
            self.model = state["model"]
            self.feature_dim = state["feature_dim"]
            self.hidden_dim = state["hidden_dim"]
            self.latent_dim = state["latent_dim"]
            self.num_layers = state["num_layers"]
            self.threshold = state["threshold"]
            self.mu = state.get("mu", 0.0)
            self.sigma = state.get("sigma", 1.0)
            self.is_fitted = True
            print(f"scikit-learn MLP Autoencoder loaded successfully from {joblib_path}")
        else:
            state = torch.load(filepath, map_location=DEVICE)
            self.feature_dim = state["feature_dim"]
            self.hidden_dim = state["hidden_dim"]
            self.latent_dim = state["latent_dim"]
            self.num_layers = state["num_layers"]
            self.threshold = state["threshold"]
            self.mu = state.get("mu", 0.0)
            self.sigma = state.get("sigma", 1.0)
            
            self.model = PyTorchLSTMAutoencoder(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_layers=self.num_layers,
                seq_len=self.seq_len
            ).to(DEVICE)
            
            self.model.load_state_dict(state["model_state"])
            self.is_fitted = True
            print(f"PyTorch LSTM Autoencoder loaded successfully from {filepath}")
