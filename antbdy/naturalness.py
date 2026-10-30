import ablang
import torch

class NaturalnessScorer:

    def __init__(self, chain="heavy"):
        print(f"Loading AbLang2 model for {chain} chain...")
        # Load a pretrained AbLang2 model for a heavy chain
        self.model = ablang.pretrained(chain=chain, iterative=False)
        self.model.freeze()

    def score_sequence(self, vh_sequence: str)->float :
        """Calculates the overall pseudo-log-likelihood score for a heavy chain sequence.

            Note: AbLang2 requires a paired format string `<VH>:<VL>`. For heavy-only
            sequences,
            we must append a trailing colon.
            """
        # Format for single-chain heavy input as per AbLang2 requirements
        formatted_seq = f"{vh_sequence}:"

        # Score the sequence using ablang
        # .score() returns the pseudo-log-likelihood
        score = self.model.score(formatted_seq)
        return float(score)



