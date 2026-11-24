class SummarizationCollator:
    def __init__(self, tokenizer, task_prefix=None, max_input_len=1024, max_target_len=64):
        self.tokenizer = tokenizer
        self.task_prefix = task_prefix
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __call__(self, batch):
        articles = [x["article"] for x in batch]
        titles = [x["title"] for x in batch]

        model_inputs = self.tokenizer(
            articles,
            max_length=self.max_input_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            text_target=titles,
            max_length=self.max_target_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        labels_ids = labels["input_ids"]
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels_ids
        return model_inputs
