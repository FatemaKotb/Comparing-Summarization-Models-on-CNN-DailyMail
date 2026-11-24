# models/model_factory.py

from models.mt5 import Mt5
from models.pegasus import Pegasus 


def get_model(model_name, gradient_checkpointing=False, pretrained_lora_weights=None):
    model_name = model_name.lower()

    if model_name == "mt5":
        wrapper = Mt5(pretrained_lora_weights=pretrained_lora_weights)
    elif model_name == "pegasus":
        wrapper = Pegasus()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    wrapper.load(gradient_checkpointing=gradient_checkpointing)
    return wrapper, wrapper.tokenizer
