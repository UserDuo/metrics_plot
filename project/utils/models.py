"""
Centralized management of models and tokenizers built on top of Hugging Face.

Responsibilities:
- Lazily load and cache AutoModel / AutoTokenizer instances to avoid repeated downloads
  and unnecessary memory usage.
- Resolve default model names from the global configuration (Config) by logical category
  (e.g. "tokenizer", "embedding", "language_model").
- Expose the current inference device ("cuda" or "cpu") in a single place.
"""
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from project.utils.config import config

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.tokenizers = {}
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def get_device(self):
        """
        Return the current inference device ("cuda" or "cpu").
        """
        return self.device

    def get_default_model_name(self, category):
        """
        Resolve the default model name for a given logical category.

        Typical categories include:
        - "tokenizer"
        - "embedding"
        - "language_model"
        """
        return config.get(f"models.{category}.default")

    def get_model(self, model_name=None, model_type="auto", category=None):
        """
        Retrieve (and cache) a model instance.

        Args:
            model_name: explicit model identifier; if omitted and category is provided,
                the default is looked up from the config under "models.<category>.default".
            model_type: one of {"auto", "causal", "sequence_classification"} to select
                the appropriate AutoModel* wrapper.
            category: logical category used to resolve a default model name.

        Returns:
            torch.nn.Module on the manager's device.
        """
        if model_name is None and category:
            model_name = self.get_default_model_name(category)
            
        if not model_name:
            raise ValueError("Model name must be provided or configured.")

        if model_name not in self.models:
            print(f"Loading model: {model_name}...")
            if model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            elif model_type == "sequence_classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            else:
                model = AutoModel.from_pretrained(model_name).to(self.device)
            self.models[model_name] = model
        return self.models[model_name]

    def get_tokenizer(self, model_name=None, category=None):
        """
        Retrieve (and cache) a tokenizer instance.

        Args:
            model_name: explicit model identifier; if omitted and category is provided,
                the default is looked up from the config under "models.<category>.default".
            category: logical category used to resolve a default model name.

        Returns:
            transformers.PreTrainedTokenizer compatible with the resolved model.
        """
        if model_name is None and category:
            model_name = self.get_default_model_name(category)
            
        if not model_name:
            raise ValueError("Model name must be provided or configured.")

        if model_name not in self.tokenizers:
            print(f"Loading tokenizer: {model_name}...")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizers[model_name]

# Global instance
model_manager = ModelManager()
