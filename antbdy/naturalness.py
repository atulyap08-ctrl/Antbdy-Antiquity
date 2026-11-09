
"""
naturalness.py

AbLang2-based antibody sequence naturalness scoring.

Features
--------
1. Single-sequence scoring:
       score_sequence()

2. Batch sequence scoring:
       score_sequences()

3. Candidate DataFrame scoring:
       score_candidates()

4. Baseline PLL and delta PLL calculation.

5. Stop-codon handling:
       Sequences containing '*' are not sent to AbLang2.

6. Duplicate protein handling:
       Identical protein sequences are scored only once.
       Scores are mapped back to all corresponding candidates.

7. Heavy-only versus paired-context comparison:
       compare_heavy_only_vs_paired()

IMPORTANT
---------
The implementation uses the AbLang2 paired checkpoint.

Heavy-only scoring means:
       [VH, ""]

Paired scoring means:
       [VH, VL]

These are different input contexts to the same checkpoint.
Heavy-only scoring does NOT mean a separate heavy-only checkpoint.

For paired scoring, VH must be first and VL second.
"""

import numpy as np
import pandas as pd
import ablang2


class NaturalnessScorer:

    VALID_CHAINS = {"heavy", "light", "paired"}

    VALID_SCORE_MODES = {
        "pseudo_log_likelihood",
        "confidence"
    }

    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    MODEL_NAME = "ablang2-paired"

    def __init__(
        self,
        chain="heavy",
        device="cpu",
        score_mode="pseudo_log_likelihood"
    ):

        chain = chain.lower()

        if chain not in self.VALID_CHAINS:
            raise ValueError(
                "chain must be 'heavy', 'light', or 'paired'."
            )

        if score_mode not in self.VALID_SCORE_MODES:
            raise ValueError(
                "score_mode must be 'pseudo_log_likelihood' "
                "or 'confidence'."
            )

        self.chain = chain
        self.device = device
        self.score_mode = score_mode

        print(
            f"Loading AbLang2 model: {self.MODEL_NAME}"
        )

        self.model = ablang2.pretrained(
            model_to_use=self.MODEL_NAME,
            random_init=False,
            ncpu=1,
            device=device
        )

        print("AbLang2 model loaded successfully.")

    # ==================================================
    # 1. SEQUENCE VALIDATION
    # ==================================================

    def _validate_sequence(self, sequence):

        if not isinstance(sequence, str):
            raise TypeError(
                "Antibody sequence must be a string."
            )

        sequence = sequence.strip().upper()

        if not sequence:
            raise ValueError(
                "Antibody sequence cannot be empty."
            )

        invalid = set(sequence) - self.VALID_AA

        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {invalid}. "
                "Only the 20 standard amino acids are supported."
            )

        return sequence

    # ==================================================
    # 2. PREPARE HEAVY-LIGHT INPUT
    # ==================================================

    def _prepare_pair(
        self,
        sequence,
        partner_sequence=None
    ):

        """
        Prepare the input in [VH, VL] order.

        chain='heavy':
            sequence = VH
            partner_sequence = optional VL

        chain='light':
            sequence = VL
            partner_sequence = optional VH

        chain='paired':
            sequence = VH
            partner_sequence = required VL
        """

        sequence = self._validate_sequence(sequence)

        if self.chain == "heavy":

            vh_sequence = sequence

            vl_sequence = (
                self._validate_sequence(partner_sequence)
                if partner_sequence is not None
                else ""
            )

        elif self.chain == "light":

            vl_sequence = sequence

            vh_sequence = (
                self._validate_sequence(partner_sequence)
                if partner_sequence is not None
                else ""
            )

        else:

            if partner_sequence is None:
                raise ValueError(
                    "Paired-chain scoring requires both VH and VL."
                )

            vh_sequence = sequence
            vl_sequence = self._validate_sequence(
                partner_sequence
            )

        # AbLang2 expects VH first and VL second.
        return [vh_sequence, vl_sequence]

    # ==================================================
    # 3. MODEL SCORING INTERNAL METHOD
    # ==================================================

    def _score_pairs(
        self,
        antibody_pairs,
        batch_size=32
    ):

        """
        Score a list of [VH, VL] inputs.

        Returns one scalar score per antibody input.
        """

        if not antibody_pairs:
            return []

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError(
                "batch_size must be a positive integer."
            )

        scores = self.model(
            antibody_pairs,
            mode=self.score_mode,
            batch_size=batch_size
        )

        scores = np.asarray(
            scores,
            dtype=float
        ).reshape(-1)

        if len(scores) != len(antibody_pairs):
            raise RuntimeError(
                "AbLang2 returned an unexpected number of scores. "
                f"Expected {len(antibody_pairs)}, "
                f"received {len(scores)}."
            )

        if not np.all(np.isfinite(scores)):
            raise RuntimeError(
                "AbLang2 returned non-finite scores."
            )

        return scores.tolist()

    # ==================================================
    # 4. BATCH SEQUENCE SCORING
    # ==================================================

    def score_sequences(
        self,
        sequences,
        partner_sequence=None,
        batch_size=32
    ):

        """
        Score multiple antibody sequences.

        Parameters
        ----------
        sequences : list of str
            List of amino-acid sequences.

        partner_sequence : str, optional
            The unchanged partner chain.

            For heavy:
                partner_sequence = VL

            For light:
                partner_sequence = VH

            For paired:
                partner_sequence = VL

        batch_size : int
            Number of sequences processed per model batch.

        Returns
        -------
        list of float
            One score per input sequence, preserving input order.
        """

        if isinstance(sequences, str):
            raise TypeError(
                "sequences must be a list or tuple of strings, "
                "not a single string."
            )

        if not isinstance(sequences, (list, tuple)):
            raise TypeError(
                "sequences must be a list or tuple."
            )

        if not sequences:
            return []

        # Validate and prepare all antibody inputs.
        antibody_pairs = [
            self._prepare_pair(
                sequence,
                partner_sequence
            )
            for sequence in sequences
        ]

        return self._score_pairs(
            antibody_pairs,
            batch_size=batch_size
        )

    # ==================================================
    # 5. SINGLE SEQUENCE SCORING
    # ==================================================

    def score_sequence(
        self,
        sequence,
        partner_sequence=None
    ):

        """
        Score one antibody sequence.

        This is a wrapper around score_sequences().
        """

        scores = self.score_sequences(
            sequences=[sequence],
            partner_sequence=partner_sequence,
            batch_size=1
        )

        return float(scores[0])

    # ==================================================
    # 6. HEAVY-ONLY VS PAIRED CONTEXT COMPARISON
    # ==================================================

    def compare_heavy_only_vs_paired(
        self,
        sequences,
        partner_sequence,
        batch_size=32
    ):

        """
        Compare heavy-only and paired-context PLL scores.

        Both contexts use the same AbLang2 paired checkpoint.

        Heavy-only:
            [VH, ""]

        Paired:
            [VH, VL]

        Parameters
        ----------
        sequences : list of str
            Heavy-chain sequences to compare.

        partner_sequence : str
            Original, unchanged light-chain sequence.

        Returns
        -------
        pandas.DataFrame

        Columns:
            sequence
            heavy_only_pll
            paired_pll
            paired_minus_heavy_only
            heavy_only_rank
            paired_rank

        Higher PLL values receive better ranks.

        NOTE:
        The difference between heavy-only and paired PLL
        reflects a change in model input context. It is not
        evidence that the antibody became biologically better.
        """

        if self.chain != "heavy":
            raise ValueError(
                "compare_heavy_only_vs_paired() requires "
                "NaturalnessScorer(chain='heavy')."
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        if not isinstance(sequences, (list, tuple)):
            raise TypeError(
                "sequences must be a list or tuple."
            )

        if not sequences:
            raise ValueError(
                "At least one sequence is required."
            )

        sequences = [
            self._validate_sequence(sequence)
            for sequence in sequences
        ]

        partner_sequence = self._validate_sequence(
            partner_sequence
        )

        # Heavy-only context: empty VL.
        heavy_only_pairs = [
            [sequence, ""]
            for sequence in sequences
        ]

        # Paired context: original VL supplied.
        paired_pairs = [
            [sequence, partner_sequence]
            for sequence in sequences
        ]

        heavy_scores = self._score_pairs(
            heavy_only_pairs,
            batch_size=batch_size
        )

        paired_scores = self._score_pairs(
            paired_pairs,
            batch_size=batch_size
        )

        comparison = pd.DataFrame({

            "sequence": sequences,

            "heavy_only_pll": heavy_scores,

            "paired_pll": paired_scores

        })

        comparison["paired_minus_heavy_only"] = (
            comparison["paired_pll"]
            - comparison["heavy_only_pll"]
        )

        # Rank 1 = highest PLL.
        comparison["heavy_only_rank"] = (
            comparison["heavy_only_pll"]
            .rank(
                ascending=False,
                method="min"
            )
        )

        comparison["paired_rank"] = (
            comparison["paired_pll"]
            .rank(
                ascending=False,
                method="min"
            )
        )

        return comparison

    # ==================================================
    # 7. SCORE MUTATION CANDIDATES
    # ==================================================

    def score_candidates(
        self,
        candidates,
        baseline_sequence,
        sequence_column="mutated_protein_sequence",
        partner_sequence=None,
        batch_size=32
    ):

        """
        Score an annotated mutation-candidate DataFrame.

        Parameters
        ----------
        candidates : pandas.DataFrame
            Output from MutationAnnotator.annotate_candidates().

        baseline_sequence : str
            Original, unmutated antibody protein sequence.

        sequence_column : str
            Column containing mutated protein sequences.

        partner_sequence : str, optional
            Unchanged partner chain.

        batch_size : int
            AbLang2 batch size.

        Returns
        -------
        pandas.DataFrame

        New columns:
            ablang2_pll
            baseline_pll
            delta_pll
            naturalness_status
            naturalness_context

        Stop-containing candidates are skipped.

        Identical protein sequences are scored once and
        their scores are mapped back to all matching rows.

        Delta PLL:
            mutant PLL - baseline PLL
        """

        # ------------------------------------------
        # A. Validate inputs
        # ------------------------------------------

        if not isinstance(candidates, pd.DataFrame):
            raise TypeError(
                "candidates must be a pandas DataFrame."
            )

        if candidates.empty:
            raise ValueError(
                "Candidate DataFrame is empty."
            )

        if sequence_column not in candidates.columns:
            raise ValueError(
                f"Column '{sequence_column}' not found."
            )

        if self.score_mode != "pseudo_log_likelihood":
            raise ValueError(
                "score_candidates() calculates PLL and delta PLL. "
                "Initialize NaturalnessScorer with "
                "score_mode='pseudo_log_likelihood'."
            )

        baseline_sequence = self._validate_sequence(
            baseline_sequence
        )

        # ------------------------------------------
        # B. Score original baseline
        # ------------------------------------------

        baseline_pll = self.score_sequence(
            baseline_sequence,
            partner_sequence=partner_sequence
        )

        # ------------------------------------------
        # C. Prepare output arrays
        # ------------------------------------------

        result = candidates.copy()

        number_of_candidates = len(result)

        candidate_scores = [
            np.nan
        ] * number_of_candidates

        statuses = [
            "not_processed"
        ] * number_of_candidates

        # Store unique valid protein sequences.
        unique_sequences = {}

        # ------------------------------------------
        # D. Identify valid and invalid candidates
        # ------------------------------------------

        for position in range(number_of_candidates):

            row = result.iloc[position]

            sequence = row[sequence_column]

            classification = row.get(
                "classification",
                ""
            )

            if pd.isna(sequence):

                statuses[position] = "missing_sequence"
                continue

            if not isinstance(sequence, str):

                raise TypeError(
                    f"Candidate at row position {position} "
                    "contains a non-string protein sequence."
                )

            sequence = sequence.strip().upper()

            if not sequence:

                statuses[position] = "missing_sequence"
                continue

            classification = (
                str(classification).strip().lower()
                if not pd.isna(classification)
                else ""
            )

            # --------------------------------------
            # STOP CODON HANDLING
            # --------------------------------------

            if (
                "*" in sequence
                or classification in {
                    "stop_gain",
                    "stop_retained"
                }
            ):

                statuses[position] = "stop_codon"
                continue

            # Validate all remaining protein sequences.
            try:

                sequence = self._validate_sequence(
                    sequence
                )

            except (ValueError, TypeError) as error:

                raise ValueError(
                    f"Invalid candidate sequence at row "
                    f"position {position}: {error}"
                ) from error

            statuses[position] = "pending"

            # --------------------------------------
            # DUPLICATE PROTEIN HANDLING
            # --------------------------------------

            if sequence not in unique_sequences:

                unique_sequences[sequence] = []

            unique_sequences[sequence].append(position)

        # ------------------------------------------
        # E. Score each unique protein only once
        # ------------------------------------------

        # If a candidate is identical to the original
        # protein, reuse its already calculated baseline PLL.

        sequences_to_score = [
            sequence
            for sequence in unique_sequences
            if sequence != baseline_sequence
        ]

        unique_scores = {}

        unique_scores[baseline_sequence] = baseline_pll

        if sequences_to_score:

            scores = self.score_sequences(
                sequences=sequences_to_score,
                partner_sequence=partner_sequence,
                batch_size=batch_size
            )

            unique_scores.update(
                dict(zip(sequences_to_score, scores))
            )

        # ------------------------------------------
        # F. Map scores back to original DataFrame
        # ------------------------------------------

        for sequence, positions in unique_sequences.items():

            score = unique_scores[sequence]

            for position in positions:

                candidate_scores[position] = score

                statuses[position] = "scored"

        # ------------------------------------------
        # G. Calculate delta PLL
        # ------------------------------------------

        delta_scores = []

        for score in candidate_scores:

            if np.isnan(score):

                delta_scores.append(np.nan)

            else:

                delta_scores.append(
                    float(score - baseline_pll)
                )

        # ------------------------------------------
        # H. Record scoring context
        # ------------------------------------------

        if self.chain == "heavy":

            context = (
                "VH+VL"
                if partner_sequence is not None
                else "VH_only"
            )

        elif self.chain == "light":

            context = (
                "VH+VL"
                if partner_sequence is not None
                else "VL_only"
            )

        else:

            context = "VH+VL"

        # ------------------------------------------
        # I. Add results to DataFrame
        # ------------------------------------------

        result["ablang2_pll"] = candidate_scores

        result["baseline_pll"] = baseline_pll

        result["delta_pll"] = delta_scores

        result["naturalness_status"] = statuses

        result["naturalness_context"] = context

        return result
