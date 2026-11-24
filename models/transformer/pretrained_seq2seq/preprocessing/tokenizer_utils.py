# preprocessing/tokenizer_utils.py

def prepare_input_text(article_text, task_prefix=None):
    """
    Prepare the input text for the model.
    For T5-style models, we optionally add a task prefix.
    """
    article_text = article_text.strip()

    if task_prefix:
        return f"{task_prefix} {article_text}"

    return article_text


def tokenize_example(
    tokenizer,
    article_text,
    headline_text=None,
    task_prefix=None,
    max_input_length=1024,
    max_output_length=15
):
    """
    Tokenize a single (article, headline) pair.

    If headline_text is None, this function can be used for inference.
    """

    # Prepare encoder input
    input_text = prepare_input_text(article_text, task_prefix)

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=max_input_length,
        padding="max_length"
    )

    # If headline exists, tokenize it as target
    if headline_text is not None:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                headline_text.strip(),
                truncation=True,
                max_length=max_output_length,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def batch_tokenize(
    tokenizer,
    articles,
    headlines=None,
    task_prefix=None,
    max_input_length=1024,
    max_output_length=15
):
    """
    Tokenize a batch of articles (and optionally headlines).
    """

    inputs = []
    for i, article in enumerate(articles):
        headline = headlines[i] if headlines is not None else None

        tokenized = tokenize_example(
            tokenizer=tokenizer,
            article_text=article,
            headline_text=headline,
            task_prefix=task_prefix,
            max_input_length=max_input_length,
            max_output_length=max_output_length
        )

        inputs.append(tokenized)

    return inputs