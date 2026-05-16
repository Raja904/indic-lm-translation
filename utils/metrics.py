import sacrebleu
from typing import List

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute BLEU-100 score using sacrebleu.
    Returns value scaled 0-100.
    """
    # sacrebleu.corpus_bleu expects a list of references lists (one list per reference set)
    # We assume one reference per hypothesis
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def compute_chrf(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute CHRF++-100 score using sacrebleu.
    Returns value scaled 0-100.
    """
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return chrf.score

def get_translation_metrics(hypotheses: List[str], references: List[str]):
    """
    Return a dictionary of BLEU and CHRF scores.
    """
    return {
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references)
    }
