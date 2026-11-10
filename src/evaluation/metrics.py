from sacrebleu.metrics import BLEU, CHRF


def compute_translation_metrics(predictions, references):
    if not predictions:
        raise ValueError("No predictions provided")

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if not all(isinstance(x, str) for x in predictions):
        raise ValueError("Predictions must be strings")

    normalized_references = []

    for ref in references:
        if isinstance(ref, str):
            ref = [ref]

        if not isinstance(ref, (list, tuple)):
            raise ValueError("Each reference must be a string or list")

        ref = [
            text.strip()
            for text in ref
            if isinstance(text, str) and text.strip()
        ]

        if not ref:
            raise ValueError("Each prediction needs a valid reference")

        normalized_references.append(ref)

    max_refs = max(len(ref) for ref in normalized_references)

    reference_streams = [
        [
            ref[i] if i < len(ref) else None
            for ref in normalized_references
        ]
        for i in range(max_refs)
    ]

    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF(word_order=2)

    bleu = bleu_metric.corpus_score(
        predictions,
        reference_streams,
    )

    chrf = chrf_metric.corpus_score(
        predictions,
        reference_streams,
    )

    return {
        "bleu": round(bleu.score, 4),
        "chrf++": round(chrf.score, 4),
        "num_examples": len(predictions),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
    }
