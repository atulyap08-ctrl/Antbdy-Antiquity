
"""
mutation_generator.py

Generate single-nucleotide substitution candidates using
pretrained ThriftySHM predictions.

Inputs:
    1. Original nucleotide sequence.
    2. ThriftySHM JSON output.

Outputs:
    A pandas DataFrame containing all three possible
    nucleotide substitutions at every genuine sequence position.

The module:
    - Removes padded positions.
    - Converts CSP logits into conditional probabilities.
    - Identifies the original nucleotide using the masked logit.
    - Generates all possible single-nucleotide substitutions.
    - Performs validation checks.

IMPORTANT:
    Mutation rates and conditional substitution probabilities
    are retained separately. No evolutionary branch length is
    assumed, and no unconditional mutation probabilities
    are calculated.


    
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class MutationGenerator:
    """Generate mutation candidates from ThriftySHM predictions."""

    BASES = ("A", "C", "G", "T")

    # ThriftySHM uses a very large negative logit to mask
    # the original nucleotide.
    MASK_THRESHOLD = -1e8

    def __init__(self, nt_sequence, shm_json):
        """
        Parameters
        ----------
        nt_sequence : str or Bio.Seq.Seq
            Original antibody nucleotide sequence.

        shm_json : str, Path or dict
            Path to the ThriftySHM JSON file, or an already
            loaded JSON object.
        """

        self.nt_sequence = str(nt_sequence).strip().upper()

        if not self.nt_sequence:
            raise ValueError("Nucleotide sequence cannot be empty.")

        invalid_bases = set(self.nt_sequence) - set(self.BASES)

        if invalid_bases:
            raise ValueError(
                f"Invalid nucleotide characters: {invalid_bases}. "
                "Only A, C, G and T are supported."
            )

        self.sequence_length = len(self.nt_sequence)

        self.shm_data = self._load_json(shm_json)

        self.rates, self.csp_logits = self._extract_predictions()

        self._remove_padding()

        self.csp_probabilities = self._calculate_csp_probabilities()

        self._validate_predictions()

    # --------------------------------------------------
    # 1. READ JSON
    # --------------------------------------------------

    @staticmethod
    def _load_json(shm_json):

        if isinstance(shm_json, dict):
            return shm_json

        path = Path(shm_json)

        if not path.is_file():
            raise FileNotFoundError(
                f"ThriftySHM JSON file not found: {path}"
            )

        with path.open("r") as file:
            return json.load(file)

    # --------------------------------------------------
    # 2. EXTRACT MUTATION RATES AND CSP LOGITS
    # --------------------------------------------------

    def _extract_predictions(self):

        if not isinstance(self.shm_data, list):
            raise ValueError(
                "Expected ThriftySHM JSON to contain a list."
            )

        if len(self.shm_data) != 2:
            raise ValueError(
                "Expected two prediction arrays: "
                "[mutation_rates, csp_logits]."
            )

        rates = np.asarray(
            self.shm_data[0],
            dtype=np.float64
        )

        logits = np.asarray(
            self.shm_data[1],
            dtype=np.float64
        )

        # Remove singleton batch dimensions.
        # Expected final shapes:
        # rates:  (padded_length,)
        # logits: (padded_length, 4)

        while rates.ndim > 1 and rates.shape[0] == 1:
            rates = rates[0]

        while logits.ndim > 2 and logits.shape[0] == 1:
            logits = logits[0]

        if rates.ndim != 1:
            raise ValueError(
                f"Unexpected mutation-rate shape: {rates.shape}"
            )

        if logits.ndim != 2 or logits.shape[1] != 4:
            raise ValueError(
                f"Unexpected CSP-logit shape: {logits.shape}. "
                "Expected (padded_length, 4)."
            )

        if len(rates) != len(logits):
            raise ValueError(
                "Mutation rates and CSP logits have "
                "different sequence lengths."
            )

        if not np.all(np.isfinite(rates)):
            raise ValueError(
                "Mutation rates contain non-finite values."
            )
        if np.any(rates < 0):
            raise ValueError("Mutation rates cannot be negative.")
        
      
        if not np.all(np.isfinite(logits)):
            raise ValueError(
                "CSP logits contain non-finite values."
            )

        return rates, logits

    # --------------------------------------------------
    # 3. REMOVE PADDING
    # --------------------------------------------------

    def _remove_padding(self):

        padded_length = len(self.rates)

        if padded_length < self.sequence_length:
            raise ValueError(
                f"ThriftySHM returned {padded_length} positions, "
                f"but the DNA sequence has "
                f"{self.sequence_length} nucleotides."
            )

        # Keep only predictions corresponding to actual
        # nucleotides in the input sequence.

        self.rates = self.rates[:self.sequence_length]

        self.csp_logits = self.csp_logits[
            :self.sequence_length
        ]

        self.padding_removed = (
            padded_length - self.sequence_length
        )

        assert len(self.rates) == self.sequence_length

        assert len(self.csp_logits) == self.sequence_length

    # --------------------------------------------------
    # 4. CONVERT CSP LOGITS INTO PROBABILITIES
    # --------------------------------------------------

    @staticmethod
    def _softmax(values):
        """
        Numerically stable softmax.

        The original nucleotide is excluded before this
        function is called.
        """

        values = np.asarray(values, dtype=np.float64)

        shifted = values - np.max(values)

        exp_values = np.exp(shifted)

        return exp_values / np.sum(exp_values)

    def _calculate_csp_probabilities(self):

        probabilities = np.zeros_like(
            self.csp_logits,
            dtype=np.float64
        )

        for position in range(self.sequence_length):

            row = self.csp_logits[position]

            # Identify the masked nucleotide.
            masked_indices = np.where(
                row <= self.MASK_THRESHOLD
            )[0]

            if len(masked_indices) != 1:
                raise ValueError(
                    f"Position {position + 1}: expected exactly "
                    f"one masked nucleotide, found "
                    f"{len(masked_indices)}."
                )

            masked_index = masked_indices[0]

            # The masked nucleotide should be the original base.
            original_base = self.BASES[masked_index]

            actual_base = self.nt_sequence[position]

            if original_base != actual_base:
                raise ValueError(
                    f"Position {position + 1}: masked nucleotide "
                    f"is {original_base}, but the input sequence "
                    f"contains {actual_base}."
                )

            # Select only the three alternative nucleotides.
            valid_indices = [
                i for i in range(4)
                if i != masked_index
            ]

            valid_logits = row[valid_indices]

            # Convert the three logits into conditional
            # substitution probabilities.
            valid_probabilities = self._softmax(valid_logits)

            probabilities[position, valid_indices] = (
                valid_probabilities
            )

        return probabilities

    # --------------------------------------------------
    # 5. VALIDATION CHECKS
    # --------------------------------------------------

    def _validate_predictions(self):

        # Check 1: Number of predictions equals sequence length.

        assert len(self.rates) == self.sequence_length

        assert len(self.csp_logits) == self.sequence_length

        # Check 2: Exactly three valid alternatives per position.

        for position in range(self.sequence_length):

            valid_probabilities = (
                self.csp_probabilities[position]
            )

            valid_count = np.count_nonzero(
                valid_probabilities > 0
            )

            assert valid_count == 3, (
                f"Position {position + 1}: "
                f"expected 3 substitution probabilities, "
                f"found {valid_count}."
            )

        # Check 3: Conditional probabilities sum to 1.

        row_sums = np.sum(
            self.csp_probabilities,
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-8
        ):
            raise ValueError(
                "CSP probabilities do not sum to 1 "
                "at every position."
            )

        # Check 4: Masked nucleotide matches original sequence.

        for position in range(self.sequence_length):

            masked_index = np.argmin(
                self.csp_logits[position]
            )

            masked_base = self.BASES[masked_index]

            original_base = self.nt_sequence[position]

            assert masked_base == original_base, (
                f"Original nucleotide mismatch at "
                f"position {position + 1}."
            )

        # Check 5: Padded positions have been excluded.

        assert len(self.rates) == self.sequence_length

        assert len(self.csp_logits) == self.sequence_length

    # --------------------------------------------------
    # 6. GENERATE MUTATION CANDIDATES
    # --------------------------------------------------

    def generate_candidates(self):

        candidates = []

        for position in range(self.sequence_length):

            original_base = self.nt_sequence[position]

            mutation_rate = float(
                self.rates[position]
            )

            for base_index, mutant_base in enumerate(self.BASES):

                # Skip the original nucleotide.
                if mutant_base == original_base:
                    continue

                csp_probability = float(
                    self.csp_probabilities[
                        position, base_index
                    ]
                )

                csp_logit = float(
                    self.csp_logits[
                        position, base_index
                    ]
                )

                candidates.append({

                    "position": position + 1,

                    "original_nt": original_base,

                    "mutant_nt": mutant_base,

                    "mutation": (
                        f"{original_base}>{mutant_base}"
                    ),

                    "mutation_rate": mutation_rate,

                    "csp_logit": csp_logit,

                    "csp_probability": csp_probability

                })

        df = pd.DataFrame(candidates)

        # --------------------------------------------------
        # FINAL DATAFRAME VALIDATION
        # --------------------------------------------------

        expected_candidates = self.sequence_length * 3

        assert len(df) == expected_candidates, (
            f"Expected {expected_candidates} candidates, "
            f"generated {len(df)}."
        )

        candidates_per_position = (
            df.groupby("position").size()
        )

        assert (
            candidates_per_position == 3
        ).all(), (
            "Some positions do not have exactly "
            "three candidate substitutions."
        )

        assert df["position"].max() == self.sequence_length

        assert df["position"].min() == 1

        # Store useful validation information.
        df.attrs["sequence_length"] = self.sequence_length

        df.attrs["padding_removed"] = self.padding_removed

        df.attrs["total_candidates"] = len(df)

        df.attrs["validation_passed"] = True

        return df
