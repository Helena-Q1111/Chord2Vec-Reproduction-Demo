"""
Hooktheory dataset preprocessing for chord2vec training.

This module extracts chord sequences from the Hooktheory.json.gz dataset
and prepares them for chord2vec training.

Hooktheory data format:
- annotations['harmony']: list of chord events
  - root_pitch_class: 0-11 (C=0, C#=1, D=2, ...)
  - root_position_intervals: defines chord quality
    - [4, 3] = major triad
    - [3, 4] = minor triad
    - [4, 3, 4] = major 7th
    - [4, 3, 3] = dominant 7th
    - etc.
  - inversion: 0, 1, 2
- annotations['keys']: list of key changes
  - tonic_pitch_class: 0-11
"""

import gzip
import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional
import pickle
import os


# Chord vocabulary constants
UNKNOWN_CHORD = "<UNK>"
PAD_CHORD = "<PAD>"

# Pitch class to note name mapping
PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Interval patterns to chord quality
INTERVAL_TO_QUALITY = {
    (4, 3): '',       # Major triad
    (3, 4): 'm',      # Minor triad
    (3, 3): 'dim',    # Diminished triad
    (4, 4): 'aug',    # Augmented triad
    (4, 3, 4): 'maj7',  # Major 7th
    (4, 3, 3): '7',     # Dominant 7th
    (3, 4, 3): 'm7',    # Minor 7th
    (3, 3, 3): 'dim7',  # Diminished 7th
    (3, 3, 4): 'm7b5',  # Half-diminished 7th
    (4, 4, 2): 'aug7',  # Augmented 7th
    (3, 4, 4): 'mMaj7', # Minor-major 7th
    (2, 5): 'sus2',     # Suspended 2nd
    (5, 2): 'sus4',     # Suspended 4th
    (4, 3, 3, 4): '9',  # Dominant 9th
    (4, 3, 4, 3): 'maj9', # Major 9th
}

# Roman numerals for scale degrees
ROMAN_NUMERALS = {
    0: 'I', 1: 'bII', 2: 'II', 3: 'bIII', 4: 'III', 5: 'IV',
    6: 'bV', 7: 'V', 8: 'bVI', 9: 'VI', 10: 'bVII', 11: 'VII'
}


def load_hooktheory_data(filepath: str) -> Dict[str, Any]:
    """Load and decompress the Hooktheory JSON dataset."""
    print(f"Loading data from {filepath}...")
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotations")
    return data


def _intervals_to_pitch_classes(intervals: Tuple[int, ...]) -> set:
    """Convert root-position interval steps to a pitch-class set including root (0)."""
    pcs = {0}
    running = 0
    for step in intervals:
        running += step
        pcs.add(running % 12)
    return pcs


def infer_quality_from_intervals(intervals: Tuple[int, ...]) -> Optional[str]:
    """
    Heuristically infer a chord quality from interval steps.

    This expands coverage to common 9/11/13 and dominant variants when
    an exact lookup table entry is missing.
    """
    if not intervals:
        return 'pedal'

    pcs = _intervals_to_pitch_classes(intervals)
    has = lambda p: (p in pcs)

    # Triad / shell identification.
    if has(3) and has(6):
        base = 'dim'
    elif has(4) and has(8):
        base = 'aug'
    elif has(3) and has(7):
        base = 'm'
    elif has(4) and has(7):
        base = ''
    elif has(5) and has(7) and not (has(3) or has(4)):
        base = 'sus4'
    elif has(2) and has(7) and not (has(3) or has(4)):
        base = 'sus2'
    elif has(7):
        base = '5'
    elif has(4):
        base = ''
    elif has(3):
        base = 'm'
    elif has(5):
        base = 'sus4'
    elif has(2):
        base = 'sus2'
    elif has(10):
        return '7no3'
    elif has(11):
        return 'maj7no3'
    elif has(1) and has(6):
        if has(9):
            return '13b9#11no3'
        return 'b9#11no3'
    elif has(6) and has(9):
        return 'add13#11no3'
    elif has(6):
        return '#11no3'
    elif has(8):
        return 'b13no3'
    elif has(1):
        return 'b9no3'
    else:
        return None

    has_b7 = has(10)
    has_maj7 = has(11)
    has9 = has(2)
    has11 = has(5)
    has13 = has(9)

    # Diminished family first.
    if base == 'dim':
        if has13:
            return 'dim7'
        if has_b7:
            return 'm7b5'
        return 'dim'

    # Suspended and power-chord families.
    if base in {'sus2', 'sus4'}:
        if has_b7:
            if has9 and has11 and has13:
                return f"13{base}"
            if has9 and has11:
                return f"11{base}"
            if has9:
                return f"9{base}"
            return f"7{base}"
        if has_maj7:
            if has9:
                return f"maj9{base}"
            return f"maj7{base}"
        if has9:
            return f"add9{base}"
        if has11:
            return f"add11{base}"
        return base

    if base == '5':
        if has_b7:
            return '75'
        return '5'

    # Minor family with extensions.
    if base == 'm':
        if has_maj7:
            if has9:
                return 'mMaj9'
            return 'mMaj7'
        if has_b7:
            if has9 and has11 and has13:
                return 'm13'
            if has9 and has11:
                return 'm11'
            if has9:
                return 'm9'
            return 'm7'
        if has9 and has11:
            return 'madd11add9'
        if has9:
            return 'madd9'
        if has11:
            return 'madd11'
        return 'm'

    # Augmented family.
    if base == 'aug':
        if has_b7:
            if has9:
                return 'aug9'
            return 'aug7'
        if has_maj7:
            return 'augMaj7'
        return 'aug'

    # Major / dominant family with optional altered tensions.
    if has_maj7:
        if has9 and has11 and has13:
            return 'maj13'
        if has9 and has11:
            return 'maj11'
        if has9:
            return 'maj9'
        return 'maj7'

    if has_b7:
        if has13:
            core = '13'
        elif has11:
            core = '11'
        elif has9:
            core = '9'
        else:
            core = '7'

        alterations = ''
        if has(1):
            alterations += 'b9'
        if has(3):
            alterations += '#9'
        if has(6):
            alterations += '#11'
        if has(8):
            alterations += 'b13'
        return core + alterations

    if has9 and has11 and has13:
        return 'add13add11add9'
    if has9 and has11:
        return 'add11add9'
    if has9:
        return 'add9'
    if has11:
        return 'add11'
    if has13:
        return 'add13'
    return ''


def build_auto_quality_map(interval_counts: Counter,
                           min_occurrence: int = 4) -> Dict[Tuple[int, ...], str]:
    """Build auto-recognized interval->quality map for frequent unknown patterns."""
    auto_map = {}
    for intervals, count in interval_counts.items():
        if count < min_occurrence:
            continue
        if intervals in INTERVAL_TO_QUALITY:
            continue
        inferred = infer_quality_from_intervals(intervals)
        if inferred is not None:
            auto_map[intervals] = inferred
    return auto_map


def intervals_to_quality(intervals: Tuple[int, ...],
                         auto_quality_map: Optional[Dict[Tuple[int, ...], str]] = None) -> Optional[str]:
    """Convert interval pattern to chord quality; fallback to auto map if available."""
    quality = INTERVAL_TO_QUALITY.get(intervals)
    if quality is not None:
        return quality
    if auto_quality_map is not None:
        return auto_quality_map.get(intervals)
    return None


def format_unknown_quality(intervals: Tuple[int, ...]) -> str:
    """Format unknown interval pattern as a stable textual tag."""
    if not intervals:
        return 'unk[empty]'
    interval_text = ','.join(str(x) for x in intervals)
    return f"unk[{interval_text}]"


def pitch_class_to_scale_degree(root_pc: int, tonic_pc: int) -> int:
    """
    Convert pitch class to scale degree relative to tonic.
    
    Args:
        root_pc: Root pitch class (0-11)
        tonic_pc: Tonic pitch class (0-11)
    
    Returns:
        Scale degree as semitones from tonic (0-11)
    """
    return (root_pc - tonic_pc) % 12


def chord_to_symbol(harmony_entry: Dict,
                    tonic_pitch_class: int,
                    use_roman: bool = True,
                    annotate_unknown: bool = True,
                    auto_quality_map: Optional[Dict[Tuple[int, ...], str]] = None) -> str:
    """
    Convert a Hooktheory harmony entry to a chord symbol.
    
    Args:
        harmony_entry: Dict with root_pitch_class, root_position_intervals, inversion
        tonic_pitch_class: The tonic pitch class for the current key
        use_roman: If True, use Roman numeral notation (I, IV, V)
                   If False, use absolute note names (C, F, G)
    
    Returns:
        Chord symbol string
    """
    root_pc = harmony_entry.get('root_pitch_class', 0)
    intervals = tuple(harmony_entry.get('root_position_intervals', [4, 3]))
    inversion = harmony_entry.get('inversion', 0)
    
    # Get chord quality from intervals
    quality = intervals_to_quality(intervals, auto_quality_map=auto_quality_map)

    if quality is None:
        if annotate_unknown:
            quality = format_unknown_quality(intervals)
        else:
            return UNKNOWN_CHORD
    
    if use_roman:
        # Convert to scale degree
        scale_degree = pitch_class_to_scale_degree(root_pc, tonic_pitch_class)
        
        # Get Roman numeral
        root_name = ROMAN_NUMERALS.get(scale_degree, str(scale_degree))
        
        # Lowercase for minor chords
        if quality in ['m', 'm7', 'dim', 'dim7', 'm7b5', 'mMaj7']:
            root_name = root_name.lower()
    else:
        # Use absolute note names
        root_name = PITCH_CLASS_NAMES[root_pc]
    
    # Build chord symbol
    chord_symbol = f"{root_name}{quality}"
    
    # Add inversion notation if needed (optional)
    # if inversion > 0:
    #     chord_symbol += f"/{inversion}"
    
    return chord_symbol


def extract_chord_symbol(chord_info: Dict, tonic_pitch_class: int = 0) -> str:
    """
    Extract a readable chord symbol from chord info.
    Wrapper for chord_to_symbol for backward compatibility.
    """
    if not chord_info:
        return UNKNOWN_CHORD
    return chord_to_symbol(chord_info, tonic_pitch_class, use_roman=True)


def extract_chord_sequences(data: Dict[str, Any], 
                           filter_audio_available: bool = False,
                           filter_melody: bool = False,
                           filter_harmony: bool = True,
                           exclude_tempo_changes: bool = False,
                           use_roman: bool = True,
                           annotate_unknown: bool = True,
                           auto_quality_min_count: int = 4) -> List[List[str]]:
    """
    Extract chord sequences from Hooktheory data.
    
    Args:
        data: Loaded Hooktheory dataset
        filter_audio_available: Only include songs with audio available
        filter_melody: Only include songs with melody annotations
        filter_harmony: Only include songs with harmony annotations (recommended)
        exclude_tempo_changes: Exclude songs with tempo changes
        use_roman: Use Roman numeral notation (True) or absolute note names (False)
    
    Returns:
        List of chord sequences (each sequence is a list of chord symbols)
    """
    chord_sequences = []
    skipped = defaultdict(int)
    total_harmony_events = 0
    interval_counter = Counter()
    selected_songs = []

    # First pass: filtering + interval frequency collection.
    for song_id, annotation in data.items():
        tags = annotation.get('tags', [])
        
        # Apply filters
        if filter_audio_available and 'AUDIO_AVAILABLE' not in tags:
            skipped['no_audio'] += 1
            continue
        
        if filter_melody and 'MELODY' not in tags:
            skipped['no_melody'] += 1
            continue
        
        if filter_harmony and 'HARMONY' not in tags:
            skipped['no_harmony'] += 1
            continue
            
        if exclude_tempo_changes and 'TEMPO_CHANGES' in tags:
            skipped['tempo_changes'] += 1
            continue
        
        # Get annotations
        annotations = annotation.get('annotations', {})
        
        # Get harmony (chord) data
        harmony = annotations.get('harmony', [])
        if not harmony:
            skipped['empty_harmony'] += 1
            continue
        
        # Get key information (tonic pitch class)
        keys = annotations.get('keys', [])
        if keys:
            tonic_pitch_class = keys[0].get('tonic_pitch_class', 0)
        else:
            tonic_pitch_class = 0  # Default to C
        
        if len(harmony) < 2:
            skipped['too_short'] += 1
            continue

        selected_songs.append((harmony, tonic_pitch_class))
        for harmony_entry in harmony:
            intervals = tuple(harmony_entry.get('root_position_intervals', [4, 3]))
            interval_counter[intervals] += 1
            total_harmony_events += 1

    auto_quality_map = build_auto_quality_map(interval_counter, min_occurrence=auto_quality_min_count)

    if auto_quality_map:
        print(f"Auto-recognized frequent unknown interval patterns (count >= {auto_quality_min_count}):")
        auto_items = sorted(auto_quality_map.items(), key=lambda x: interval_counter[x[0]], reverse=True)
        for intervals, quality in auto_items[:25]:
            print(f"  {intervals} -> {quality} (count={interval_counter[intervals]})")

    unknown_interval_counter = Counter()
    auto_recovered_events = 0

    # Second pass: symbol conversion with auto quality map.
    for harmony, tonic_pitch_class in selected_songs:
        chords = []
        for harmony_entry in harmony:
            intervals = tuple(harmony_entry.get('root_position_intervals', [4, 3]))

            base_quality = INTERVAL_TO_QUALITY.get(intervals)
            final_quality = intervals_to_quality(intervals, auto_quality_map=auto_quality_map)
            if base_quality is None and final_quality is not None:
                auto_recovered_events += 1
            if final_quality is None:
                unknown_interval_counter[intervals] += 1

            chord_symbol = chord_to_symbol(
                harmony_entry,
                tonic_pitch_class,
                use_roman=use_roman,
                annotate_unknown=annotate_unknown,
                auto_quality_map=auto_quality_map,
            )
            if chord_symbol and chord_symbol != UNKNOWN_CHORD:
                chords.append(chord_symbol)

        if len(chords) >= 2:
            chord_sequences.append(chords)
    
    print(f"\nExtracted {len(chord_sequences)} chord sequences")
    print(f"Skipped: {dict(skipped)}")

    known_events = total_harmony_events - sum(unknown_interval_counter.values())
    coverage = (known_events / total_harmony_events) if total_harmony_events else 0.0
    base_unknown_events = sum(
        count for intervals, count in interval_counter.items() if intervals not in INTERVAL_TO_QUALITY
    )
    base_known_events = total_harmony_events - base_unknown_events
    base_coverage = (base_known_events / total_harmony_events) if total_harmony_events else 0.0
    print(f"Total harmony events: {total_harmony_events}")
    print(f"Base known interval events: {base_known_events} ({base_coverage:.2%})")
    print(f"Known interval events after auto-map: {known_events} ({coverage:.2%})")
    print(f"Auto-recovered events: {auto_recovered_events}")
    print(f"Unknown interval events after auto-map: {sum(unknown_interval_counter.values())} ({1.0-coverage:.2%})")

    if unknown_interval_counter:
        print("Top unknown interval patterns:")
        for intervals, count in unknown_interval_counter.most_common(15):
            print(f"  {intervals}: {count}")
    
    # Print some sample sequences
    if chord_sequences:
        print("\nSample sequences:")
        for i, seq in enumerate(chord_sequences[:5]):
            print(f"  {i+1}: {' -> '.join(seq[:10])}{'...' if len(seq) > 10 else ''}")
    
    return chord_sequences


def audit_hooktheory_chord_coverage(data: Dict[str, Any],
                                    auto_quality_min_count: int = 4) -> Dict[str, Any]:
    """Audit interval-pattern coverage for Hooktheory harmony events."""
    total = 0
    known = 0
    unknown_counter = Counter()
    known_counter = Counter()
    interval_counter = Counter()

    for annotation in data.values():
        harmony = annotation.get('annotations', {}).get('harmony') or []
        for h in harmony:
            total += 1
            intervals = tuple(h.get('root_position_intervals', [4, 3]))
            interval_counter[intervals] += 1
            quality = INTERVAL_TO_QUALITY.get(intervals)
            if quality is None:
                unknown_counter[intervals] += 1
            else:
                known += 1
                known_counter[quality] += 1

    auto_quality_map = build_auto_quality_map(interval_counter, min_occurrence=auto_quality_min_count)
    auto_recovered_events = sum(interval_counter[k] for k in auto_quality_map.keys())
    known_after_auto = known + auto_recovered_events
    unknown_after_auto = total - known_after_auto

    unknown = total - known
    report = {
        'total_harmony_events': total,
        'known_harmony_events': known,
        'unknown_harmony_events': unknown,
        'known_coverage': (known / total) if total else 0.0,
        'unknown_coverage': (unknown / total) if total else 0.0,
        'auto_quality_min_count': auto_quality_min_count,
        'auto_recovered_harmony_events': auto_recovered_events,
        'known_harmony_events_after_auto': known_after_auto,
        'unknown_harmony_events_after_auto': unknown_after_auto,
        'known_coverage_after_auto': (known_after_auto / total) if total else 0.0,
        'unknown_coverage_after_auto': (unknown_after_auto / total) if total else 0.0,
        'known_quality_counts': dict(known_counter),
        'auto_interval_to_quality': {
            str(k): v for k, v in sorted(auto_quality_map.items(), key=lambda kv: interval_counter[kv[0]], reverse=True)
        },
        'auto_interval_counts': {
            str(k): interval_counter[k] for k in sorted(auto_quality_map.keys(), key=lambda x: interval_counter[x], reverse=True)
        },
        'unknown_interval_counts': {
            str(k): v for k, v in unknown_counter.most_common()
        },
    }
    return report


def build_vocabulary(chord_sequences: List[List[str]], 
                    min_count: int = 2) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build chord vocabulary from sequences.
    
    Args:
        chord_sequences: List of chord sequences
        min_count: Minimum occurrence count to include in vocabulary
    
    Returns:
        chord2idx: Mapping from chord symbol to index
        idx2chord: Mapping from index to chord symbol
    """
    # Count chord occurrences
    chord_counts = Counter()
    for seq in chord_sequences:
        chord_counts.update(seq)
    
    # Filter by minimum count and create vocabulary
    chord2idx = {PAD_CHORD: 0, UNKNOWN_CHORD: 1}
    
    for chord, count in chord_counts.most_common():
        if count >= min_count:
            chord2idx[chord] = len(chord2idx)
    
    idx2chord = {v: k for k, v in chord2idx.items()}
    
    print(f"\nVocabulary size: {len(chord2idx)}")
    print(f"Top 20 chords: {chord_counts.most_common(20)}")
    
    return chord2idx, idx2chord


def generate_skipgram_pairs(chord_sequences: List[List[str]],
                            chord2idx: Dict[str, int],
                            window_size: int = 2,
                            subsample_threshold: float = 1e-4) -> List[Tuple[int, int]]:
    """
    Generate skip-gram training pairs (center chord, context chord).
    
    Uses subsampling for frequent chords (like word2vec) to balance training:
    p(keep) = sqrt(threshold / frequency) if frequency > threshold, else 1.0
    
    Args:
        chord_sequences: List of chord sequences
        chord2idx: Chord to index mapping
        window_size: Number of chords to consider on each side
        subsample_threshold: Threshold for subsampling frequent chords (0 to disable)
    
    Returns:
        List of (center_idx, context_idx) pairs
    """
    pairs = []
    unk_idx = chord2idx.get(UNKNOWN_CHORD, 1)
    
    # Calculate chord frequencies for subsampling
    total_chords = 0
    chord_counts = Counter()
    for seq in chord_sequences:
        for chord in seq:
            if chord in chord2idx and chord2idx[chord] != unk_idx:
                chord_counts[chord] += 1
                total_chords += 1
    
    # Calculate discard probabilities
    discard_probs = {}
    if subsample_threshold > 0 and total_chords > 0:
        for chord, count in chord_counts.items():
            freq = count / total_chords
            if freq > subsample_threshold:
                # Probability to KEEP the word
                keep_prob = np.sqrt(subsample_threshold / freq)
                discard_probs[chord] = 1.0 - keep_prob
            else:
                discard_probs[chord] = 0.0
        
        # Log subsampling info
        high_freq_chords = [(c, p) for c, p in discard_probs.items() if p > 0.1]
        if high_freq_chords:
            high_freq_chords.sort(key=lambda x: x[1], reverse=True)
            print(f"\nSubsampling high-frequency chords (top 10):")
            for chord, prob in high_freq_chords[:10]:
                print(f"  {chord}: discard prob = {prob:.3f}")
    
    for seq in chord_sequences:
        # Convert to indices (with possible subsampling)
        kept_positions = []
        seq_indices = []
        for i, chord in enumerate(seq):
            idx = chord2idx.get(chord, unk_idx)
            if idx == unk_idx:
                continue
            
            # Subsample frequent chords
            if subsample_threshold > 0 and chord in discard_probs:
                if np.random.random() < discard_probs[chord]:
                    continue  # Discard this chord
            
            kept_positions.append(i)
            seq_indices.append(idx)
        
        # Generate pairs from kept chords
        for i, center_idx in enumerate(seq_indices):
            # Use dynamic window based on original positions
            original_pos = kept_positions[i]
            
            for j, context_idx in enumerate(seq_indices):
                if i != j:
                    context_original_pos = kept_positions[j]
                    # Check if within window based on original positions
                    if abs(original_pos - context_original_pos) <= window_size:
                        pairs.append((center_idx, context_idx))
    
    print(f"\nGenerated {len(pairs)} skip-gram pairs")
    return pairs


def split_data(chord_sequences: List[List[str]], 
               train_ratio: float = 0.8,
               valid_ratio: float = 0.1,
               seed: int = 42) -> Tuple[List, List, List]:
    """Split chord sequences into train/valid/test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(chord_sequences))
    
    n_train = int(len(chord_sequences) * train_ratio)
    n_valid = int(len(chord_sequences) * valid_ratio)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    
    train_seqs = [chord_sequences[i] for i in train_idx]
    valid_seqs = [chord_sequences[i] for i in valid_idx]
    test_seqs = [chord_sequences[i] for i in test_idx]
    
    print(f"\nData split: train={len(train_seqs)}, valid={len(valid_seqs)}, test={len(test_seqs)}")
    
    return train_seqs, valid_seqs, test_seqs


def prepare_dataset(hooktheory_path: str,
                   output_dir: str = "processed",
                   window_size: int = 2,
                   min_chord_count: int = 2,
                   use_roman: bool = True) -> Optional[Dict[str, Any]]:
    """
    Complete data preparation pipeline.
    
    Args:
        hooktheory_path: Path to Hooktheory.json.gz
        output_dir: Directory to save processed files
        window_size: Context window size for skip-gram
        min_chord_count: Minimum chord count for vocabulary
        use_roman: Use Roman numeral notation (True) or absolute note names (False)
    
    Returns:
        Dictionary with all processed data and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_hooktheory_data(hooktheory_path)
    
    # Extract chord sequences using the correct Hooktheory format
    chord_sequences = extract_chord_sequences(
        data, 
        filter_audio_available=False,  # Include all annotations
        filter_melody=False,
        filter_harmony=True,  # Only include songs with harmony data
        exclude_tempo_changes=False,
        use_roman=use_roman
    )
    
    if not chord_sequences:
        print("ERROR: No chord sequences extracted from data!")
        print("Please check the data format.")
        return None
    
    # Build vocabulary
    chord2idx, idx2chord = build_vocabulary(chord_sequences, min_chord_count)
    
    # Split data
    train_seqs, valid_seqs, test_seqs = split_data(chord_sequences)
    
    # Generate skip-gram pairs for each split
    train_pairs = generate_skipgram_pairs(train_seqs, chord2idx, window_size)
    valid_pairs = generate_skipgram_pairs(valid_seqs, chord2idx, window_size)
    test_pairs = generate_skipgram_pairs(test_seqs, chord2idx, window_size)
    
    # Create result dictionary
    result = {
        'chord2idx': chord2idx,
        'idx2chord': idx2chord,
        'vocab_size': len(chord2idx),
        'train_pairs': train_pairs,
        'valid_pairs': valid_pairs,
        'test_pairs': test_pairs,
        'train_sequences': train_seqs,
        'valid_sequences': valid_seqs,
        'test_sequences': test_seqs,
        'window_size': window_size,
    }
    
    # Save processed data
    output_path = os.path.join(output_dir, 'chord2vec_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nSaved processed data to {output_path}")
    
    # Save vocabulary separately for easy access
    vocab_path = os.path.join(output_dir, 'vocabulary.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'chord2idx': chord2idx, 'idx2chord': {str(k): v for k, v in idx2chord.items()}}, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    return result


def load_processed_data(filepath: str) -> Dict[str, Any]:
    """Load previously processed data."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    
    # Default path
    hooktheory_path = "Hooktheory.json.gz"
    
    if len(sys.argv) > 1:
        hooktheory_path = sys.argv[1]
    
    # Run preprocessing
    result = prepare_dataset(hooktheory_path)

    if result is None:
        print("\nData preparation failed.")
        raise SystemExit(1)

    print("\n" + "="*50)
    print("Data preparation complete!")
    print(f"Vocabulary size: {result['vocab_size']}")
    print(f"Training pairs: {len(result['train_pairs'])}")
    print(f"Validation pairs: {len(result['valid_pairs'])}")
    print(f"Test pairs: {len(result['test_pairs'])}")
