import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import copy
import itertools
import logging

from src.dasproject.data import DASDataset
from src.dasproject.model import ConvAutoencoder
from src.dasproject.utils import get_device

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0
    for data in dataloader:
        img, _ = data
        img = img.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Evaluates model on validation set."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def run_grid_search(config):
    """
    Orchestrates the training pipeline with Grid Search.
    
    Returns:
        best_model: The model with lowest validation loss.
        best_params: Dict of parameters used for that model.
    """
    device = get_device()
    print(f"Device: {device}")

    # 1. Prepare Data
    raw_path = Path(config['paths']['raw_data_path'])
    # Only pick files that are NOT in the inference folder
    train_files = list(raw_path.glob("*.hdf5"))
    
    print(f"Found {len(train_files)} training files.")
    dataset = DASDataset(train_files, config, mode='train')
    
    train_size = int(config['data']['train_val_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False)

    # 2. Grid Search Setup
    lrs = config['hyperparameter_search']['learning_rates']
    latents = config['hyperparameter_search']['latent_dims']
    best_val_loss = float('inf')
    best_model_state = None
    best_params = {}

    # 3. Search Loop
    for lr, latent_dim in itertools.product(lrs, latents):
        print(f"\n--- Training: LR={lr}, Latent={latent_dim} ---")
        
        model = ConvAutoencoder(config, latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training Loop
        for epoch in range(config['training']['epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
        # Check if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_params = {'lr': lr, 'latent_dim': latent_dim}
            print(f"--> New Best Model found (Val Loss: {val_loss:.4f})")
            logging.getLogger(__name__).info(f"New best model: LR={lr}, Latent={latent_dim}, Val Loss={val_loss:.4f}")
    # 4. Save Best Model
    save_path = Path(config['paths']['model_path']) / "best_cae.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_model = ConvAutoencoder(config, best_params['latent_dim'])
    final_model.load_state_dict(best_model_state)
    torch.save(final_model.state_dict(), save_path)
    print(f"\nBest model saved to {save_path} with params: {best_params}")
    logging.getLogger(__name__).info(f"Best model saved to {save_path} with params: {best_params}")
    
    return final_model, best_params