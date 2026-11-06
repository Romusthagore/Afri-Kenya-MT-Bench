from sacrebleu.metrics import BLEU, CHRF


def compute_mt_metrics(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if not predictions:
        raise ValueError("Empty predictions")

    chrf = CHRF(word_order=2)
    bleu = BLEU()

    return {
        "chrf": chrf.corpus_score(predictions, [references]).score,
        "bleu": bleu.corpus_score(predictions, [references]).score,
    }
