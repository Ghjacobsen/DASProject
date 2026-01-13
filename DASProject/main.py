import logging
import torch
from pathlib import Path
from src.dasproject.utils import load_config, setup_logging
from src.dasproject.train import run_grid_search
from src.dasproject.evaluate import run_inference_and_save  # Changed Name
from src.dasproject.visualize import generate_reports         # Changed Name
from src.dasproject.model import ConvAutoencoder

def main():
    # 1. Setup
    config = load_config("config.yaml")
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("--- Pipeline Started ---")

    # 2. Train (Unchanged)
    logger.info("Step 1: Training & Grid Search")
    best_model, best_params = run_grid_search(config)
    
    # 3. Inference (Save to HDF5)
    logger.info("Step 2: Running Inference & Stitching")
    
    # Load best weights freshly to ensure validity
    model_path = Path(config['paths']['model_path']) / "best_cae.pth"
    loaded_model = ConvAutoencoder(config, best_params['latent_dim'])
    loaded_model.load_state_dict(torch.load(model_path))
    
    # This now saves files to data/reconstructions
    run_inference_and_save(loaded_model, config)
    
    # 4. Visualize (Load HDF5 -> Plot)
    logger.info("Step 3: Generating Visual Reports")
    generate_reports(config)
    
    logger.info("✅ Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()