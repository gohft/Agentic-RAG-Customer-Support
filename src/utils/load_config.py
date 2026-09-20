from omegaconf import OmegaConf

def load_config(path: str) -> dict:
    """
    Load configuration file and return as dictionary. 
    """
    config = OmegaConf.load(path)
    if not config:
        raise ValueError(f"Config file at {path} is empty or invalid.")
    # return config in dictionary, resolve environment vars as well 
    
    return OmegaConf.to_container(config, resolve=True)