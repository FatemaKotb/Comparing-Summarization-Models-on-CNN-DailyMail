# models/base_model.py

import torch
from abc import ABC, abstractmethod
#ignore 


class BaseHeadlineModel(ABC):
    def __init__(self, model_name, tokenizer_name=None, pretrained_lora_weights=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.pretrained_lora_weights = pretrained_lora_weights

        self.model = None
        self.tokenizer = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load(self):
        raise NotImplementedError

    def move_to_device(self):
        if self.model is None:
            raise RuntimeError("Model must be loaded before moving to device.")
        self.model.to(self.device)

    def enable_gradient_checkpointing(self):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=15,
        num_beams=4,
        length_penalty=1.2,
        no_repeat_ngram_size=3
    ):
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True
            )
            
    def save(self, output_dir):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Nothing to save. Load the model first.")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)