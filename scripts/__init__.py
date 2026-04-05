"""
Chord2Vec - Learning chord embeddings from music data.

This package provides tools for training chord2vec models using the Hooktheory dataset,
inspired by the word2vec architecture but adapted for musical chord sequences.
"""

from .model import (
    SkipGramChord2Vec,
    LinearChord2Vec,
    Seq2SeqChord2Vec,
    ChordDataset,
    Chord2VecTrainer,
    compute_analogy
)

from .data_processing import (
    load_hooktheory_data,
    extract_chord_sequences,
    build_vocabulary,
    generate_skipgram_pairs,
    prepare_dataset,
    load_processed_data
)

__version__ = '0.1.0'
__all__ = [
    'SkipGramChord2Vec',
    'LinearChord2Vec', 
    'Seq2SeqChord2Vec',
    'ChordDataset',
    'Chord2VecTrainer',
    'compute_analogy',
    'load_hooktheory_data',
    'extract_chord_sequences',
    'build_vocabulary',
    'generate_skipgram_pairs',
    'prepare_dataset',
    'load_processed_data',
]
