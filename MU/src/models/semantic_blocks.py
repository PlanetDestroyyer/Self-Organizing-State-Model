"""Semantic block definitions for MU matrix structure"""

from typing import Tuple, List


class SemanticBlockLayout:
    """Defines the 16 semantic blocks in 8×8 matrix"""

    BLOCKS = {
        # Name: (row_start, col_start, row_end, col_end, description)
        'I': (0, 0, 2, 2, 'Identity - core token meaning'),
        'S': (0, 2, 2, 4, 'Structure - grammatical properties'),
        'C1': (0, 4, 2, 6, 'Context-Local - immediate context'),
        'C2': (0, 6, 2, 8, 'Context-Global - document context'),
        'R1': (2, 0, 4, 2, 'Relations-Syntactic - syntax dependencies'),
        'R2': (2, 2, 4, 4, 'Relations-Semantic - meaning dependencies'),
        'T': (2, 4, 4, 6, 'Transformation - compositional changes'),
        'K': (2, 6, 4, 8, 'Knowledge - world knowledge grounding'),
        'G': (4, 0, 6, 2, 'Global - document coherence'),
        'M': (4, 2, 6, 4, 'Modality - certainty/tense/mood'),
        'D': (4, 4, 6, 6, 'Discourse - rhetorical structure'),
        'F': (4, 6, 6, 8, 'Frame - semantic frame roles'),
        'P': (6, 0, 8, 2, 'Position - positional encoding'),
        'E': (6, 2, 8, 4, 'Entity - named entity properties'),
        'A': (6, 4, 8, 6, 'Affect - sentiment/emotion'),
        'X': (6, 6, 8, 8, 'Extension - flexible/learned purpose'),
    }

    @classmethod
    def get_block_indices(cls, block_name: str) -> Tuple[int, int, int, int]:
        """Get (r1, c1, r2, c2) for a semantic block"""
        return cls.BLOCKS[block_name][:4]

    @classmethod
    def get_all_block_names(cls) -> List[str]:
        """Get list of all block names"""
        return list(cls.BLOCKS.keys())
