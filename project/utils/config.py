"""
Configuration manager for the research pipeline.

Responsibilities:
- Load project-wide settings from a YAML file (or fall back to built-in defaults).
- Expose configuration values via dot-separated paths, e.g. "models.tokenizer.default".
- Use a singleton pattern so that all modules observe a consistent configuration state.
"""
import yaml
import os

class Config:
    _instance = None
    
    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance.load(config_path)
        return cls._instance
    
    def load(self, config_path=None):
        """
        Load configuration from disk or use an internal default.

        Priority:
        1) If config_path is provided, read from that path.
        2) Otherwise, read project/config/default.yaml relative to this file.
        3) If the file does not exist, fall back to a minimal built-in config
           that still allows the pipeline to run.
        """
        if config_path is None:
            # Default to project/config/default.yaml
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "default.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            self._config = {
                "models": {
                    "tokenizer": {"default": "bert-base-uncased"},
                    "embedding": {"default": "sentence-transformers/all-MiniLM-L6-v2"},
                    "language_model": {"default": "gpt2"}
                }
            }
            
    def get(self, path, default=None):
        """
        Retrieve a configuration value using a dot-separated path.

        Example:
            get("models.tokenizer.default")

        If any part of the path is missing, return the provided default value.
        """
        keys = path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

# Global instance
config = Config()
