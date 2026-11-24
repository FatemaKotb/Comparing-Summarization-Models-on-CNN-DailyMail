from rouge_score import rouge_scorer
from typing import List, Dict


class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        assert len(predictions) == len(references)

        totals = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }
        
        predictions = [pred.strip() for pred in predictions]
        references = [ref.strip() for ref in references]
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            totals["rouge1"] += scores["rouge1"].fmeasure
            totals["rouge2"] += scores["rouge2"].fmeasure
            totals["rougeL"] += scores["rougeL"].fmeasure

        n = len(predictions)
        return {k: v / n for k, v in totals.items()}
    
if __name__ == "__main__":
    print("If you wanna test lowkey")
    