# models/mt5_model.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .base_model import BaseHeadlineModel
from peft import PeftModel

class Mt5(BaseHeadlineModel):
    def __init__(self, model_name="google/mt5-base", tokenizer_name=None, pretrained_lora_weights=None):
        super().__init__(model_name, tokenizer_name, pretrained_lora_weights)

    def load(self, gradient_checkpointing=False):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        print("not none please : ", self.pretrained_lora_weights is not None)
        if self.pretrained_lora_weights is not None:
            self.model = PeftModel.from_pretrained(self.model, self.pretrained_lora_weights)
            print("LoRA layers added to the model.")

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

        self.move_to_device()

        return self