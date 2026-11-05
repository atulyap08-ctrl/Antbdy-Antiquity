
"""
mutation_annotator.py

Annotates single-nucleotide substitution candidates and
determines their effects on the translated protein.

Inputs:
    1. Original nucleotide sequence (5' -> 3' coding strand).
    2. mutation_candidates DataFrame from MutationGenerator.

Outputs:
    An annotated pandas DataFrame containing:
        - Original and mutated nucleotide information
        - Original and mutated codons
        - Original and mutated amino acids
        - Amino-acid substitution notation
        - Mutation classification
        - Complete mutated DNA sequence
        - Complete mutated protein sequence

Supported classifications:
    - synonymous
    - missense
    - stop_gain
    - stop_loss
    - stop_retained

Reading frame:
    frame_offset=0 means the first nucleotide is the first
    nucleotide of a complete codon.

IMPORTANT:
    The input must already be in the 5' -> 3' coding orientation.
    This module does not infer biological strand orientation.

 -Incomplete terminal codons: The default is to reject them. Only use incomplete_codon="trim" when you know the 
trailing nucleotides are outside the coding region.

    
"""

import pandas as pd

from Bio.Seq import Seq


class MutationAnnotator:

    VALID_BASES = set("ACGT")

    REQUIRED_COLUMNS = {
        "position",
        "original_nt",
        "mutant_nt",
        "mutation",
        "mutation_rate",
        "csp_logit",
        "csp_probability"
    }

    VALID_CLASSIFICATIONS = {
        "synonymous",
        "missense",
        "stop_gain",
        "stop_loss",
        "stop_retained"
    }

    def __init__(
        self,
        nt_sequence,
        mutation_candidates,
        frame_offset=0,
        sequence_orientation="coding_5to3",
        incomplete_codon="error",
        allow_terminal_stop=True
    ):
        """
        Parameters
        ----------
        nt_sequence : str or Bio.Seq.Seq
            Original DNA sequence.

        mutation_candidates : pandas.DataFrame
            Output from MutationGenerator.generate_candidates().

        frame_offset : int
            Number of leading nucleotides before the first
            complete codon. Must be 0, 1 or 2.

        sequence_orientation : str
            Must be 'coding_5to3'.

        incomplete_codon : str
            'error': Reject incomplete terminal codons.
            'trim': Exclude incomplete terminal nucleotides
                    from translation.

        allow_terminal_stop : bool
            Whether a terminal stop codon is allowed.
        """

        # --------------------------------------------
        # 1. Store and validate inputs
        # --------------------------------------------

        self.nt_sequence = str(nt_sequence).strip().upper()

        self.candidates = mutation_candidates.copy()

        self.frame_offset = frame_offset

        self.sequence_orientation = sequence_orientation

        self.incomplete_codon = incomplete_codon

        self.allow_terminal_stop = allow_terminal_stop

        self._validate_inputs()

        self.sequence_length = len(self.nt_sequence)

        # --------------------------------------------
        # 2. Establish complete coding region
        # --------------------------------------------

        self.coding_sequence = self.nt_sequence[
            self.frame_offset:
        ]

        remainder = len(self.coding_sequence) % 3

        if remainder:

            if self.incomplete_codon == "error":

                raise ValueError(
                    f"Sequence contains {remainder} "
                    "incomplete terminal nucleotide(s). "
                    "Use incomplete_codon='trim' only "
                    "if the trailing bases are known to "
                    "be outside the coding region."
                )

            self.trailing_bases = (
                self.coding_sequence[-remainder:]
            )

            self.complete_coding_sequence = (
                self.coding_sequence[:-remainder]
            )

        else:

            self.trailing_bases = ""

            self.complete_coding_sequence = (
                self.coding_sequence
            )

        if not self.complete_coding_sequence:

            raise ValueError(
                "No complete codons available for translation."
            )

        # --------------------------------------------
        # 3. Translate original protein
        # --------------------------------------------

        self.original_protein = str(
            Seq(self.complete_coding_sequence).translate()
        )

        self._validate_original_protein()

    # ==================================================
    # INPUT VALIDATION
    # ==================================================

    def _validate_inputs(self):

        # Check nucleotide sequence.

        if not self.nt_sequence:

            raise ValueError(
                "Nucleotide sequence cannot be empty."
            )

        invalid_bases = (
            set(self.nt_sequence) - self.VALID_BASES
        )

        if invalid_bases:

            raise ValueError(
                f"Invalid nucleotide characters: "
                f"{invalid_bases}. "
                "Only A, C, G and T are supported."
            )

        # Check orientation declaration.

        if self.sequence_orientation != "coding_5to3":

            raise ValueError(
                "Expected a 5' -> 3' coding-strand sequence. "
                "Reverse-complement the sequence and regenerate "
                "ThriftySHM predictions before annotation if "
                "the supplied sequence is on the opposite strand."
            )

        # Check reading frame.

        if (
            not isinstance(self.frame_offset, int)
            or isinstance(self.frame_offset, bool)
            or self.frame_offset not in (0, 1, 2)
        ):

            raise ValueError(
                "frame_offset must be 0, 1 or 2."
            )

        # Check incomplete-codon handling.

        if self.incomplete_codon not in ("error", "trim"):

            raise ValueError(
                "incomplete_codon must be 'error' or 'trim'."
            )

        # Check DataFrame.

        if not isinstance(self.candidates, pd.DataFrame):

            raise TypeError(
                "mutation_candidates must be a pandas DataFrame."
            )

        missing_columns = (
            self.REQUIRED_COLUMNS - set(self.candidates.columns)
        )

        if missing_columns:

            raise ValueError(
                f"Missing required DataFrame columns: "
                f"{missing_columns}"
            )

        if self.candidates.empty:

            raise ValueError(
                "Mutation candidate DataFrame is empty."
            )

        if self.candidates[list(self.REQUIRED_COLUMNS)].isnull().any().any():

            raise ValueError(
                "Mutation candidates contain missing values."
            )

    # ==================================================
    # ORIGINAL PROTEIN VALIDATION
    # ==================================================

    def _validate_original_protein(self):

        if "*" in self.original_protein:

            stop_positions = [
                i + 1
                for i, aa in enumerate(self.original_protein)
                if aa == "*"
            ]

            internal_stops = stop_positions

            # A terminal stop codon is acceptable if enabled.
            if (
                self.allow_terminal_stop
                and stop_positions
                and stop_positions[-1] == len(self.original_protein)
            ):

                internal_stops = stop_positions[:-1]

            if internal_stops:

                raise ValueError(
                    "Original sequence contains internal "
                    f"stop codon(s) at amino-acid position(s): "
                    f"{internal_stops}."
                )

            if (
                not self.allow_terminal_stop
                and stop_positions
            ):

                raise ValueError(
                    "Original sequence contains a stop codon."
                )

    # ==================================================
    # CODON IDENTIFICATION
    # ==================================================

    def _get_codon_information(self, position):

        """
        Identify the codon containing a 1-based nucleotide position.
        """

        zero_based_position = position - 1

        coding_position = (
            zero_based_position - self.frame_offset
        )

        if coding_position < 0:

            raise ValueError(
                f"Position {position} lies before the "
                "declared coding frame."
            )

        coding_length = len(
            self.complete_coding_sequence
        )

        if coding_position >= coding_length:

            raise ValueError(
                f"Position {position} belongs to an "
                "incomplete terminal codon or lies outside "
                "the complete coding region."
            )

        # Find the beginning of the relevant codon.

        codon_start = (
            self.frame_offset
            + (coding_position // 3) * 3
        )

        codon_end = codon_start + 3

        original_codon = self.nt_sequence[
            codon_start:codon_end
        ]

        # Position within the codon: 0, 1 or 2.

        codon_offset = (
            zero_based_position - codon_start
        )

        original_aa = str(
            Seq(original_codon).translate()
        )

        # Amino-acid position, 1-based.

        aa_position = (
            (codon_start - self.frame_offset) // 3
        ) + 1

        return {
            "codon_start": codon_start,
            "codon_end": codon_end,
            "codon_offset": codon_offset,
            "original_codon": original_codon,
            "original_aa": original_aa,
            "aa_position": aa_position
        }

    # ==================================================
    # CLASSIFY MUTATION
    # ==================================================

    @staticmethod
    def _classify_mutation(original_aa, mutant_aa):

        if original_aa == "*" and mutant_aa == "*":

            return "stop_retained"

        if original_aa != "*" and mutant_aa == "*":

            return "stop_gain"

        if original_aa == "*" and mutant_aa != "*":

            return "stop_loss"

        if original_aa == mutant_aa:

            return "synonymous"

        return "missense"

    # ==================================================
    # ANNOTATE A SINGLE CANDIDATE
    # ==================================================

    def _annotate_single_candidate(self, candidate):

        position = int(candidate["position"])

        original_nt = str(
            candidate["original_nt"]
        ).upper()

        mutant_nt = str(
            candidate["mutant_nt"]
        ).upper()

        # Validate nucleotide position.

        if not 1 <= position <= self.sequence_length:

            raise ValueError(
                f"Invalid nucleotide position: {position}."
            )

        actual_original_nt = (
            self.nt_sequence[position - 1]
        )

        if original_nt != actual_original_nt:

            raise ValueError(
                f"Position {position}: DataFrame reports "
                f"{original_nt}, but original sequence contains "
                f"{actual_original_nt}."
            )

        if original_nt not in self.VALID_BASES:

            raise ValueError(
                f"Invalid original nucleotide at position {position}."
            )

        if mutant_nt not in self.VALID_BASES:

            raise ValueError(
                f"Invalid mutant nucleotide at position {position}."
            )

        if mutant_nt == original_nt:

            raise ValueError(
                f"Position {position}: mutant nucleotide "
                "is identical to original nucleotide."
            )

        # Validate mutation notation.

        expected_mutation = (
            f"{original_nt}>{mutant_nt}"
        )

        if str(candidate["mutation"]) != expected_mutation:

            raise ValueError(
                f"Position {position}: mutation notation "
                f"should be {expected_mutation}."
            )

        # Identify original codon.

        codon_info = self._get_codon_information(position)

        codon_start = codon_info["codon_start"]

        codon_end = codon_info["codon_end"]

        original_codon = codon_info["original_codon"]

        codon_offset = codon_info["codon_offset"]

        original_aa = codon_info["original_aa"]

        # Generate mutated codon.

        mutated_codon_list = list(original_codon)

        mutated_codon_list[codon_offset] = mutant_nt

        mutated_codon = "".join(
            mutated_codon_list
        )

        # Translate both codons.

        mutant_aa = str(
            Seq(mutated_codon).translate()
        )

        # Classify mutation.

        classification = self._classify_mutation(
            original_aa,
            mutant_aa
        )

        # Generate complete mutated DNA sequence.

        mutated_nt_list = list(self.nt_sequence)

        mutated_nt_list[position - 1] = mutant_nt

        mutated_nt_sequence = "".join(
            mutated_nt_list
        )

        # Verify exactly one nucleotide changed.

        differences = sum(
            a != b
            for a, b in zip(
                self.nt_sequence,
                mutated_nt_sequence
            )
        )

        if differences != 1:

            raise RuntimeError(
                f"Expected exactly one nucleotide difference "
                f"at position {position}, found {differences}."
            )

        # Verify sequence length is unchanged.

        if len(mutated_nt_sequence) != self.sequence_length:

            raise RuntimeError(
                "Mutation unexpectedly changed sequence length."
            )

        # Translate complete mutant protein.

        mutant_coding_sequence = mutated_nt_sequence[
            self.frame_offset:
        ]

        mutant_coding_sequence = (
            mutant_coding_sequence[
                :len(self.complete_coding_sequence)
            ]
        )

        mutated_protein_sequence = str(
            Seq(mutant_coding_sequence).translate()
        )

        # Verify that only the expected amino-acid position
        # has changed.

        expected_protein = list(self.original_protein)

        expected_protein[
            codon_info["aa_position"] - 1
        ] = mutant_aa

        if "".join(expected_protein) != mutated_protein_sequence:

            raise RuntimeError(
                f"Protein translation mismatch at nucleotide "
                f"position {position}."
            )

        # Amino-acid substitution notation.

        aa_change = (
            f"{original_aa}{codon_info['aa_position']}"
            f"{mutant_aa}"
        )

        return {

            "codon_position": codon_info["aa_position"],

            "codon_start_nt": codon_start + 1,

            "original_codon": original_codon,

            "mutated_codon": mutated_codon,

            "original_aa": original_aa,

            "mutant_aa": mutant_aa,

            "aa_change": aa_change,

            "classification": classification,

            "mutated_nt_sequence": mutated_nt_sequence,

            "mutated_protein_sequence": mutated_protein_sequence

        }

    # ==================================================
    # ANNOTATE ALL MUTATION CANDIDATES
    # ==================================================

    def annotate_candidates(self):

        annotated_candidates = []

        for _, candidate in self.candidates.iterrows():

            annotation = self._annotate_single_candidate(
                candidate
            )

            annotated_candidates.append(annotation)

        annotation_df = pd.DataFrame(
            annotated_candidates,
            index=self.candidates.index
        )

        # Combine original ThriftySHM information
        # with new protein-level annotations.

        result_df = pd.concat(
            [
                self.candidates.reset_index(drop=True),
                annotation_df.reset_index(drop=True)
            ],
            axis=1
        )

        # ==================================================
        # FINAL VALIDATION
        # ==================================================

        expected_rows = len(self.candidates)

        assert len(result_df) == expected_rows

        # Every candidate must produce exactly one
        # nucleotide substitution.

        assert (
            result_df["mutated_nt_sequence"]
            .str.len()
            .eq(self.sequence_length)
            .all()
        )

        # All mutant proteins must have the expected length.

        assert (
            result_df["mutated_protein_sequence"]
            .str.len()
            .eq(len(self.original_protein))
            .all()
        )

        # Check that only the expected classifications exist.

        observed_classifications = set(
            result_df["classification"]
        )

        assert observed_classifications.issubset(
            self.VALID_CLASSIFICATIONS
        )

        # Store metadata.

        result_df.attrs["original_nt_sequence"] = (
            self.nt_sequence
        )

        result_df.attrs["original_protein_sequence"] = (
            self.original_protein
        )

        result_df.attrs["frame_offset"] = (
            self.frame_offset
        )

        result_df.attrs["sequence_length"] = (
            self.sequence_length
        )

        result_df.attrs["protein_length"] = (
            len(self.original_protein)
        )

        result_df.attrs["trailing_bases_excluded"] = (
            self.trailing_bases
        )

        result_df.attrs["total_candidates"] = (
            len(result_df)
        )

        result_df.attrs["validation_passed"] = True

        return result_df
