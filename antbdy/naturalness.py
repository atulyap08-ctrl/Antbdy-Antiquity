
import numpy as np
import ablang2


class NaturalnessScorer:
    """
    AbLang2-based antibody sequence scorer.

    Supports:
        - Heavy chain only
        - Light chain only
        - Paired heavy + light chains

    The default scoring method is pseudo_log_likelihood.
    """

    VALID_CHAINS = {"heavy", "light", "paired"}
    VALID_SCORE_MODES = {"pseudo_log_likelihood", "confidence"}
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        chain="heavy",
        device="cpu",
        score_mode="pseudo_log_likelihood"
    ):
        chain = chain.lower()

        if chain not in self.VALID_CHAINS:
            raise ValueError(
                "chain must be 'heavy', 'light', or 'paired'"
            )

        if score_mode not in self.VALID_SCORE_MODES:
            raise ValueError(
                "score_mode must be 'pseudo_log_likelihood' "
                "or 'confidence'"
            )

        self.chain = chain
        self.score_mode = score_mode

        print("Loading AbLang2 model...")

        self.model = ablang2.pretrained(
            model_to_use="ablang2-paired",
            random_init=False,
            ncpu=1,
            device=device
        )

        print("AbLang2 model loaded successfully.")

    def _validate_sequence(self, sequence):
        sequence = sequence.strip().upper()

        if not sequence:
            raise ValueError("Antibody sequence cannot be empty.")

        invalid = set(sequence) - self.VALID_AA

        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {invalid}"
            )

        return sequence

    def score_sequence(self, sequence, partner_sequence=None):
        """
        Calculate an AbLang2 sequence score.

        For chain='heavy':
            sequence = VH
            partner_sequence = optional VL

        For chain='light':
            sequence = VL
            partner_sequence = optional VH

        For chain='paired':
            sequence = VH
            partner_sequence = VL
        """

        sequence = self._validate_sequence(sequence)

        if self.chain == "heavy":
            vh_sequence = sequence
            vl_sequence = (
                self._validate_sequence(partner_sequence)
                if partner_sequence else ""
            )

        elif self.chain == "light":
            vl_sequence = sequence
            vh_sequence = (
                self._validate_sequence(partner_sequence)
                if partner_sequence else ""
            )

        else:
            if not partner_sequence:
                raise ValueError(
                    "Paired-chain scoring requires both VH and VL."
                )

            vh_sequence = sequence
            vl_sequence = self._validate_sequence(partner_sequence)

        # AbLang2 expects each antibody as [VH, VL].
        antibody = [[vh_sequence, vl_sequence]]

        scores = self.model(
            antibody,
            mode=self.score_mode
        )

        # Extract the single sequence-level score.
        score = np.asarray(scores, dtype=float).reshape(-1)

        if score.size != 1:
            raise RuntimeError(
                f"Expected one score, received {score.size}."
            )

        return float(score[0])
