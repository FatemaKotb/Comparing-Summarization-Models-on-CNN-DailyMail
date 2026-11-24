from rouge_score import rouge_scorer

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = max(1, len(preds))
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # (target, prediction)
        for k in agg:
            agg[k] += s[k].fmeasure
    for k in agg:
        agg[k] /= n
    return agg
