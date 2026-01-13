import yaml
from pathlib import Path
import logging
import sys

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_project_root():
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent

def setup_logging(config):
    """
    Configures the root logger based on config settings.
    Creates the log directory if it doesn't exist.
    """
    # 1. Parse config
    log_level_str = config['logging']['level'].upper()
    log_file_path = Path(config['logging']['file'])
    
    # 2. Create directory if missing (Automating your directory question)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. specific numeric level
    level = getattr(logging, log_level_str, logging.INFO)

    # 4. Configure format
    # Format: [Time] [Level] [Module]: Message
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    # 5. Reset handlers if they exist (prevents duplicate logs in notebooks/reruns)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # 6. Add Handlers
    # Handler A: File (Saves to logs/pipeline.log)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Handler B: Stream (Prints to console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    
    # Apply settings
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    logging.info(f"Logging set up. Writing to {log_file_path}")