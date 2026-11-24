# models/pegasus.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .base_model import BaseHeadlineModel
from peft import PeftModel, prepare_model_for_kbit_training

class Pegasus(BaseHeadlineModel):
    def __init__(self, model_name="google/pegasus-cnn_dailymail", tokenizer_name=None, pretrained_lora_weights=None):
        super().__init__(model_name, tokenizer_name, pretrained_lora_weights)

    def load(self, gradient_checkpointing=False):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name or self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.pretrained_lora_weights is not None:
            # Load PEFT LoRA weights
            self.model = PeftModel.from_pretrained(self.model, self.pretrained_lora_weights)
            print("LoRA layers added to the model.")
        else:
            # Prepare base model for LoRA training (freezes base, enables grad checkpointing if applicable)
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=gradient_checkpointing)

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

        self.move_to_device()
        return self