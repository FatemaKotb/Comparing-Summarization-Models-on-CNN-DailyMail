import sys
from pathlib import Path
import json
import argparse
import torch

# ---------------------------
# Add project root to sys.path
# ---------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.model_factory import get_model

# ---------------------------
# Sample article (DailyMail style)
# ---------------------------
sample_article = (
    '''
        The European Central Bank decided on Thursday to keep interest rates unchanged,
        citing persistent inflationary pressures and ongoing economic uncertainty across the eurozone.
        ECB President Christine Lagarde said the governing council would continue to monitor incoming data closely before making any policy adjustments. Markets reacted cautiously to the announcement, with the euro weakening slightly against the dollar while government bond yields remained stable. Analysts remain divided on whether a rate cut could occur later this year.
    '''
)

def main():
    argParser = argparse.ArgumentParser(description="Generate a summary using a pre-trained model.")
    argParser.add_argument("--config", type=str, default="configs/Mt5.json", help="Path to the configuration JSON file.")
    args = argParser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        params = json.load(f)
        
    model_name = params["model_name"]
    task_prefix = params.get("task_prefix", "")
    task_suffix = params.get("task_suffix", "")
    print(f"Loading {model_name} model...")
    model_wrapper, tokenizer = get_model(model_name, gradient_checkpointing=False, pretrained_lora_weights=params.get("pretrained_lora_weights", None))
    model = model_wrapper.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # for task_prefix in prefixes:
    input_text = f"{task_prefix} {sample_article} {task_suffix}".strip()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=params["max_tokens"]).to(device)
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        length_penalty=1.0
    )

    headline = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated headline: {headline}")

if __name__ == "__main__":
    main()
