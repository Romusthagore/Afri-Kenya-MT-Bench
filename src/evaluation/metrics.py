import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


def compute_translation_metrics(predictions, references):
    if not predictions:
        raise ValueError("No predictions provided")

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if not all(isinstance(x, str) for x in predictions + references):
        raise ValueError("Predictions and references must be strings")

    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)

    bleu = bleu_metric.corpus_score(
        predictions,
        [references],
    )

    chrf = chrf_metric.corpus_score(
        predictions,
        [references],
    )

    return {
        "bleu": round(bleu.score, 4),
        "chrf++": round(chrf.score, 4),
        "num_examples": len(predictions),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
    }
