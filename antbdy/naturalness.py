import ablang
import numpy as np
import torch


class NaturalnessScorer:
  """Wrapper class to load AbLang and score antibody sequence naturalness."""

  def __init__(self, chain="heavy"):
    print(f"Loading AbLang model for {chain} chain...")
    self.model = ablang.pretrained(chain=chain)
    self.model.freeze()

  def score_sequence(self, vh_sequence: str) -> float:
    """Calculates the naturalness likelihood score for a heavy chain amino acid sequence."""
    # Pass the clean amino acid sequence directly
    likelihoods = self.model([vh_sequence], mode="likelihood")

    score_val = likelihoods[0]
    if hasattr(score_val, "mean"):
      return float(score_val.mean())
    elif isinstance(score_val, (list, tuple, np.ndarray)):
      return float(np.mean(score_val))
    return float(score_val)
