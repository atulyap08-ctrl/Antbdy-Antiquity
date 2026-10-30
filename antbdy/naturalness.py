import ablang
import torch


class NaturalnessScorer:
  """Wrapper class to load AbLang2 and score antibody sequence naturalness."""

  def __init__(self, chain="heavy"):
    print(f"Loading AbLang2 model for {chain} chain...")
    self.model = ablang.pretrained(chain=chain)
    self.model.freeze()

  def score_sequence(self, vh_sequence: str) -> float:
    """Calculates the pseudo-log-likelihood score for a heavy chain sequence."""
    formatted_seq = f"{vh_sequence}:"

    # Call the model directly with mode='pseudo_log_likelihood'
    score = self.model([formatted_seq], mode="pseudo_log_likelihood")

    # Handle array or scalar return types safely
    if hasattr(score, "__getitem__"):
      return float(score[0])
    return float(score)
