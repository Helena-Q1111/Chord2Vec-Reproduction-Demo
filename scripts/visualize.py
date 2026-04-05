"""
Visualization and evaluation tools for chord2vec.

Includes:
- t-SNE visualization of chord embeddings
- Chord similarity analysis
- Chord analogy evaluation
- Chord progression clustering
"""

import argparse
import csv
import json
import os
import pickle
import re
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

# Conditional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization features disabled.")

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Some features disabled.")

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Spearman correlation disabled.")

from model import SkipGramChord2Vec, compute_analogy


ENHARMONIC_TO_SHARP = {
    'Cb': 'B',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
    'E#': 'F',
    'B#': 'C',
}


def _normalize_note_token(token: str) -> str:
    """Normalize note token (supports both Db and bD styles) to sharp naming."""
    if token.startswith('b') and len(token) >= 2 and token[1] in 'ABCDEFG':
        token = token[1] + 'b'
    return ENHARMONIC_TO_SHARP.get(token, token)


def _parse_absolute_chord_symbol(chord: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse absolute chord symbol into (normalized_root, quality_suffix)."""
    if chord in {'<PAD>', '<UNK>'}:
        return None, None

    if chord.startswith('b') and len(chord) >= 2 and chord[1] in 'ABCDEFG':
        root_token = chord[:2]
        quality = chord[2:]
    else:
        m = re.match(r'^([A-G](?:#|b)?)(.*)$', chord)
        if not m:
            return None, None
        root_token = m.group(1)
        quality = m.group(2)

    return _normalize_note_token(root_token), quality


def _quality_to_family(quality: str) -> Optional[str]:
    """Map chord quality suffix to one of major/minor/dim/dominant families."""
    if quality.startswith('unk['):
        return None

    dominant_like = (
        quality == '7' or quality == '9' or quality == '11' or quality == '13'
        or quality.startswith('7') or quality.startswith('9') or quality.startswith('13')
        or quality == 'aug7'
    )
    if dominant_like and not quality.startswith('maj') and not quality.startswith('m'):
        return 'dominant'

    if quality in {'dim', 'dim7', 'm7b5'}:
        return 'dim'

    if quality in {'m', 'm7', 'mMaj7'} or (quality.startswith('m') and not quality.startswith('maj')):
        return 'minor'

    # Everything else is grouped into major-family by design:
    # major triads, maj7/maj9, sus2/sus4, augmented, etc.
    return 'major'


def plot_multiring_chord_correlation(
        embeddings: np.ndarray,
        idx2chord: Dict[int, str],
        output_path: str = 'multiring_chord_correlation.png',
        edge_csv_path: str = 'multiring_edge_correlations.csv',
        edge_json_path: str = 'multiring_edge_correlations.json') -> Dict[str, Any]:
    """
    Draw a multi-ring chord relation graph and color arrows by cosine similarity.

    The outer alternating order follows the requested correction around B:
    ... B (major) -> G#m (minor) -> F# (major) -> bEm (minor) -> Db (major) ...
    """
    if not HAS_MATPLOTLIB:
        print("Multi-ring graph requires matplotlib")
        return {}

    major_cycle = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    minor_cycle = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'bEm', 'bBm', 'Fm', 'Cm', 'Gm', 'Dm']
    dominant_cycle = ['C7', 'G7', 'D7', 'A7', 'E7', 'B7', 'F#7', 'Db7', 'Ab7', 'Eb7', 'Bb7', 'F7']
    dim_cycle = ['Cdim', 'Gdim', 'Ddim', 'Adim', 'Edim', 'Bdim', 'F#dim', 'Dbdim', 'Abdim', 'Ebdim', 'Bbdim', 'Fdim']

    # User-requested ring rotation:
    # dominant ring: counterclockwise 2 steps.
    # diminished ring: clockwise 1 step from previous setting (was ccw=5, now ccw=4).
    dominant_rotation_ccw = 2
    dim_rotation_ccw = 4

    # Build family buckets from available vocabulary chords.
    family_root_to_indices = {}
    for i, chord in idx2chord.items():
        root, quality = _parse_absolute_chord_symbol(chord)
        if root is None or quality is None:
            continue
        family = _quality_to_family(quality)
        if family is None:
            continue
        key = (family, root)
        family_root_to_indices.setdefault(key, []).append(i)

    def centroid_for(family: str, display_label: str):
        base = display_label
        if family == 'minor' and base.endswith('m'):
            base = base[:-1]
        if family.startswith('dominant') and base.endswith('7'):
            base = base[:-1]
        if family == 'dim' and base.endswith('dim'):
            base = base[:-3]

        root_norm = _normalize_note_token(base)
        lookup_family = 'dominant' if family.startswith('dominant') else family
        indices = family_root_to_indices.get((lookup_family, root_norm), [])
        if not indices:
            return None
        return embeddings[indices].mean(axis=0)

    node_records = []
    for i in range(12):
        node_records.append(('major', major_cycle[i]))
        node_records.append(('minor', minor_cycle[i]))

    # 24-node outer ring (alternating major/minor)
    outer_radius = 1.30
    outer_positions = {}
    for i, (family, label) in enumerate(node_records):
        theta = np.pi / 2 - 2 * np.pi * i / len(node_records)
        outer_positions[(family, label)] = (outer_radius * np.cos(theta), outer_radius * np.sin(theta))

    # Inner rings: dominant and diminished.
    dominant_radius = 0.82
    dim_radius = 0.50
    dominant_positions = {}
    dim_positions = {}

    for i, label in enumerate(dominant_cycle):
        theta = np.pi / 2 - 2 * np.pi * (i - dominant_rotation_ccw) / 12
        dominant_positions[('dominant', label)] = (
            dominant_radius * np.cos(theta),
            dominant_radius * np.sin(theta)
        )
    for i, label in enumerate(dim_cycle):
        theta = np.pi / 2 - 2 * np.pi * (i - dim_rotation_ccw) / 12
        dim_positions[('dim', label)] = (dim_radius * np.cos(theta), dim_radius * np.sin(theta))

    node_positions = {}
    node_positions.update(outer_positions)
    node_positions.update(dominant_positions)
    node_positions.update(dim_positions)

    node_embeddings = {}
    for key in node_positions:
        family, label = key
        node_embeddings[key] = centroid_for(family, label)

    def cosine_sim(v1, v2):
        if v1 is None or v2 is None:
            return None
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return None
        return float(np.dot(v1, v2) / (n1 * n2))

    edges = []

    # Outer ring adjacency (clockwise)
    for i in range(len(node_records)):
        src = node_records[i]
        dst = node_records[(i + 1) % len(node_records)]
        edges.append((src, dst, 'outer_clockwise'))

    # Dominant ring adjacency
    for i in range(12):
        src = ('dominant', dominant_cycle[i])
        dst = ('dominant', dominant_cycle[(i + 1) % 12])
        edges.append((src, dst, 'dominant_clockwise'))

    # Dim ring adjacency
    for i in range(12):
        src = ('dim', dim_cycle[i])
        dst = ('dim', dim_cycle[(i + 1) % 12])
        edges.append((src, dst, 'dim_clockwise'))

    # Cross-family links with requested functional chain:
    # dim -> dominant -> minor -> dominant -> major
    # Example: Edim -> A7 -> Dm -> G7 -> C.
    for i in range(12):
        m = ('major', major_cycle[i])
        n = ('minor', minor_cycle[i])
        d7_to_minor = ('dominant', dominant_cycle[(i + 4) % 12])
        d7_to_major = ('dominant', dominant_cycle[(i + 1) % 12])
        dd = ('dim', dim_cycle[i])
        d7_for_dim = ('dominant', dominant_cycle[(i - 1) % 12])
        edges.append((m, n, 'major_to_minor'))
        edges.append((n, d7_to_major, 'minor_to_dominant'))
        edges.append((d7_to_minor, n, 'dominant_to_minor'))
        edges.append((d7_to_major, m, 'dominant_to_major'))
        edges.append((dd, d7_for_dim, 'dim_to_dominant'))

    edge_rows = []
    sims = []
    for src, dst, relation in edges:
        sim = cosine_sim(node_embeddings.get(src), node_embeddings.get(dst))
        edge_rows.append({
            'source': f"{src[1]} ({src[0]})",
            'target': f"{dst[1]} ({dst[0]})",
            'relation': relation,
            'cosine_similarity': sim,
        })
        if sim is not None:
            sims.append(sim)

    if not sims:
        print("No valid edge similarities found. Check whether model was trained with absolute notation.")
        return {}

    # Fixed color scale requested by user: 0.5 -> 1.0
    # Red -> yellow gradient.
    sim_min = 0.5
    sim_max = 1.0
    norm = plt.Normalize(sim_min, sim_max)
    cmap = cm.get_cmap('autumn')

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    family_node_color = {
        'major': '#00c853',
        'minor': '#40c4ff',
        'dominant': '#ff1744',
        'dim': '#d500f9',
    }

    # Draw edges first
    for row, (src, dst, _) in zip(edge_rows, edges):
        sim = row['cosine_similarity']
        if sim is None:
            continue
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[dst]
        lw = 0.9 + 2.2 * np.clip(norm(sim), 0.0, 1.0)
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color=cmap(norm(sim)),
                lw=lw,
                alpha=0.82,
                shrinkA=12,
                shrinkB=12,
                connectionstyle='arc3,rad=0.09',
            ),
            zorder=2,
        )

    # Draw nodes
    for (family, label), (x, y) in node_positions.items():
        emb = node_embeddings[(family, label)]
        is_available = emb is not None
        fill = family_node_color[family] if is_available else '#424242'
        ax.scatter([x], [y], s=900 if family in {'major', 'minor'} else 620,
                   c=fill, edgecolors='white', linewidths=1.6, zorder=4)
        ax.text(x, y, label, color='white', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Cosine similarity on arrow', color='white')
    cbar.set_ticks(np.arange(0.5, 1.01, 0.05))
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_title('Chord Multi-Ring Correlation Graph', color='white', fontsize=18, pad=18)
    ax.set_aspect('equal')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved multi-ring graph to {output_path}")
    plt.close(fig)

    # Save edge correlations for explicit per-arrow validation.
    with open(edge_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'target', 'relation', 'cosine_similarity'])
        writer.writeheader()
        writer.writerows(edge_rows)
    print(f"Saved edge correlation table to {edge_csv_path}")

    with open(edge_json_path, 'w', encoding='utf-8') as f:
        json.dump(edge_rows, f, indent=2, ensure_ascii=False)
    print(f"Saved edge correlation json to {edge_json_path}")

    available_nodes = sum(1 for v in node_embeddings.values() if v is not None)
    return {
        'available_nodes': available_nodes,
        'total_nodes': len(node_embeddings),
        'edge_count': len(edge_rows),
        'valid_edge_count': int(sum(1 for r in edge_rows if r['cosine_similarity'] is not None)),
        'outer_sequence': [f"{fam}:{lab}" for fam, lab in node_records],
    }


def load_model_and_vocab(model_dir: str):
    """Load trained model and vocabulary."""
    # Load vocabulary
    vocab_path = os.path.join(model_dir, 'vocabulary.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    chord2idx = vocab_data['chord2idx']
    idx2chord = {int(k): v for k, v in vocab_data['idx2chord'].items()}
    
    # Load model
    model_path = os.path.join(model_dir, 'best_model.pt')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = SkipGramChord2Vec(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load embeddings
    embedding_path = os.path.join(model_dir, 'chord_embeddings.npy')
    embeddings = np.load(embedding_path)
    
    return model, chord2idx, idx2chord, embeddings


def visualize_embeddings_tsne(embeddings: np.ndarray,
                               idx2chord: Dict[int, str],
                               output_path: str = 'chord_embeddings_tsne.png',
                               perplexity: int = 30,
                               n_iter: int = 1000,
                               top_n: Optional[int] = 100,
                               highlight_chords: Optional[List[str]] = None):
    """
    Visualize chord embeddings using t-SNE.
    
    Args:
        embeddings: Chord embedding matrix [vocab_size, embedding_dim]
        idx2chord: Index to chord mapping
        output_path: Path to save visualization
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        top_n: Only plot top N most frequent chords (None for all)
        highlight_chords: List of chord symbols to highlight
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Visualization requires matplotlib and sklearn")
        return
    
    print(f"Computing t-SNE projection (perplexity={perplexity}, n_iter={n_iter})...")
    
    # Select subset of chords to visualize
    if top_n and top_n < len(embeddings):
        # Assume indices are ordered by frequency (0, 1 are PAD, UNK)
        indices = list(range(2, min(top_n + 2, len(embeddings))))
    else:
        indices = list(range(len(embeddings)))
    
    subset_embeddings = embeddings[indices]
    subset_labels = [idx2chord.get(i, f"UNK_{i}") for i in indices]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(indices) - 1),
                n_iter=n_iter, random_state=42)
    coords = tsne.fit_transform(subset_embeddings)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Determine colors for different chord types
    colors = []
    for label in subset_labels:
        if label.startswith('I') and not label.startswith('II'):
            colors.append('red')
        elif label.startswith('IV'):
            colors.append('blue')
        elif label.startswith('V'):
            colors.append('green')
        elif label.startswith('VI') or label.startswith('vi'):
            colors.append('orange')
        elif label.startswith('II') or label.startswith('ii'):
            colors.append('purple')
        elif label.startswith('III') or label.startswith('iii'):
            colors.append('cyan')
        elif label.startswith('VII') or label.startswith('vii'):
            colors.append('brown')
        else:
            colors.append('gray')
    
    # Plot points
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=50)
    
    # Add labels
    for i, label in enumerate(subset_labels):
        # Highlight specific chords
        if highlight_chords and label in highlight_chords:
            plt.annotate(label, (coords[i, 0], coords[i, 1]),
                        fontsize=12, fontweight='bold', color='black',
                        xytext=(5, 5), textcoords='offset points')
        else:
            plt.annotate(label, (coords[i, 0], coords[i, 1]),
                        fontsize=8, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
    
    plt.title('Chord2Vec Embeddings (t-SNE Projection)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='I (Tonic)'),
        Patch(facecolor='blue', label='IV (Subdominant)'),
        Patch(facecolor='green', label='V (Dominant)'),
        Patch(facecolor='orange', label='VI'),
        Patch(facecolor='purple', label='II'),
        Patch(facecolor='cyan', label='III'),
        Patch(facecolor='brown', label='VII'),
        Patch(facecolor='gray', label='Other'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {output_path}")
    plt.close()


def analyze_chord_similarities(model: SkipGramChord2Vec,
                                chord2idx: Dict[str, int],
                                idx2chord: Dict[int, str],
                                query_chords: Optional[List[str]] = None,
                                top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze chord similarities for given query chords.
    
    Args:
        model: Trained chord2vec model
        chord2idx: Chord to index mapping
        idx2chord: Index to chord mapping
        query_chords: List of chords to analyze (default: common chords)
        top_k: Number of similar chords to return
    
    Returns:
        Dictionary mapping chord to list of (similar_chord, similarity) tuples
    """
    if query_chords is None:
        # Default to common chords
        query_chords = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII',
                       'Im', 'IVm', 'Vm', 'I7', 'IV7', 'V7']
    
    results = {}
    
    for chord in query_chords:
        if chord in chord2idx:
            idx = chord2idx[chord]
            similar = model.most_similar(idx, idx2chord, top_k=top_k)
            results[chord] = similar
    
    return results


def evaluate_analogies(model: SkipGramChord2Vec,
                       chord2idx: Dict[str, int],
                       idx2chord: Dict[int, str],
                       analogies: Optional[List[Tuple[str, str, str, str]]] = None) -> Dict:
    """
    Evaluate chord analogies.
    
    Format: a:b :: c:d means "a is to b as c is to d"
    
    Args:
        model: Trained chord2vec model
        chord2idx: Chord to index mapping
        idx2chord: Index to chord mapping
        analogies: List of (a, b, c, expected_d) tuples
    
    Returns:
        Evaluation results
    """
    if analogies is None:
        # Default musical analogies
        # These test if the model learned functional harmony relationships
        analogies = [
            # Tonic-Dominant relationships
            ('I', 'V', 'IV', 'I'),  # I->V as IV->I (resolution)
            ('I', 'IV', 'V', 'I'),  # I->IV as V->I (plagal vs authentic)
            
            # Relative major/minor (if model has minor chords)
            ('I', 'VI', 'IV', 'II'),
            
            # Circle of fifths
            ('I', 'IV', 'IV', 'VII'),
            ('V', 'I', 'II', 'V'),
        ]
    
    results = {
        'correct': 0,
        'total': 0,
        'details': []
    }
    
    for a, b, c, expected in analogies:
        if not all(chord in chord2idx for chord in [a, b, c]):
            continue
        
        predictions = compute_analogy(model, idx2chord, chord2idx, a, b, c, top_k=5)
        
        if predictions:
            top_prediction = predictions[0][0]
            is_correct = (top_prediction == expected)
            
            results['details'].append({
                'analogy': f"{a}:{b} :: {c}:?",
                'expected': expected,
                'predicted': top_prediction,
                'correct': is_correct,
                'top_5': predictions
            })
            
            if is_correct:
                results['correct'] += 1
            results['total'] += 1
    
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
    else:
        results['accuracy'] = 0.0
    
    return results


def cluster_chords(embeddings: np.ndarray,
                   idx2chord: Dict[int, str],
                   n_clusters: int = 8,
                   output_path: str = 'chord_clusters.json') -> Dict:
    """
    Cluster chords based on their embeddings.
    
    Args:
        embeddings: Chord embedding matrix
        idx2chord: Index to chord mapping
        n_clusters: Number of clusters
        output_path: Path to save clustering results
    
    Returns:
        Clustering results
    """
    if not HAS_SKLEARN:
        print("Clustering requires sklearn")
        return {}
    
    print(f"Clustering chords into {n_clusters} groups...")
    
    # Skip PAD and UNK tokens
    valid_indices = list(range(2, len(embeddings)))
    valid_embeddings = embeddings[valid_indices]
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_embeddings)
    
    # Group chords by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for i, cluster_id in enumerate(cluster_labels):
        chord_idx = valid_indices[i]
        chord_name = idx2chord.get(chord_idx, f"UNK_{chord_idx}")
        clusters[cluster_id].append(chord_name)
    
    # Calculate cluster statistics
    results = {
        'n_clusters': n_clusters,
        'clusters': {str(k): v for k, v in clusters.items()}
    }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved clustering results to {output_path}")
    
    # Print summary
    print("\nCluster summary:")
    for cluster_id, chords in clusters.items():
        chord_preview = ', '.join(chords[:10])
        if len(chords) > 10:
            chord_preview += f"... (+{len(chords)-10} more)"
        print(f"  Cluster {cluster_id}: {chord_preview}")
    
    return results


def plot_training_history(history_path: str, output_path: str = 'training_history.png'):
    """Plot training history."""
    if not HAS_MATPLOTLIB:
        print("Plotting requires matplotlib")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    valid_losses = history['valid_losses']
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    
    best_epoch = history.get('best_epoch', valid_losses.index(min(valid_losses)) + 1)
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Chord2Vec Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved training history plot to {output_path}")
    plt.close()


def compute_chord_similarity_matrix(embeddings: np.ndarray,
                                     idx2chord: Dict[int, str],
                                     chord_list: Optional[List[str]] = None,
                                     output_path: str = 'similarity_matrix.png'):
    """
    Compute and visualize chord similarity matrix.
    
    Args:
        embeddings: Chord embedding matrix
        idx2chord: Index to chord mapping
        chord_list: List of chords to include (None for default)
        output_path: Path to save visualization
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Similarity matrix requires matplotlib and sklearn")
        return
    
    chord2idx = {v: k for k, v in idx2chord.items()}
    
    if chord_list is None:
        # Use common chords
        chord_list = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    
    # Filter to existing chords
    chord_list = [c for c in chord_list if c in chord2idx]
    
    if len(chord_list) < 2:
        print("Not enough chords found for similarity matrix")
        return
    
    # Get embeddings for selected chords
    indices = [chord2idx[c] for c in chord_list]
    selected_embeddings = embeddings[indices]
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels
    plt.xticks(range(len(chord_list)), chord_list, rotation=45, ha='right')
    plt.yticks(range(len(chord_list)), chord_list)
    
    # Add values
    for i in range(len(chord_list)):
        for j in range(len(chord_list)):
            plt.text(j, i, f'{sim_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=8)
    
    plt.title('Chord Similarity Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved similarity matrix to {output_path}")
    plt.close()


def analyze_circle_of_fifths(embeddings: np.ndarray,
                              idx2chord: Dict[int, str],
                              output_path: str = 'circle_of_fifths.png'):
    """
    Analyze if the learned embeddings capture circle of fifths relationships.
    
    The circle of fifths (in Roman numerals for major chords):
    I → V → II → VI → III → VII → bV → bII → bVI → bIII → bVII → IV → I
    
    Each step up is a perfect fifth (7 semitones), and adjacent chords
    in the circle should have similar functional relationships.
    
    Args:
        embeddings: Chord embedding matrix
        idx2chord: Index to chord mapping
        output_path: Path to save visualization
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Circle of fifths analysis requires matplotlib and sklearn")
        return {}
    
    chord2idx = {v: k for k, v in idx2chord.items()}
    
    # Circle of fifths progression (major chords)
    circle_major = ['I', 'V', 'II', 'VI', 'III', 'VII', 'bV', 'bII', 'bVI', 'bIII', 'bVII', 'IV']
    # Circle of fifths (minor chords)
    circle_minor = ['im', 'vm', 'iim', 'vim', 'iiim', 'viim', 'bvm', 'biim', 'bvim', 'biiim', 'bviim', 'ivm']
    
    print("\n" + "="*60)
    print("Circle of Fifths Analysis")
    print("="*60)
    
    def analyze_circle(circle, name):
        """Analyze one circle of fifths."""
        # Get available chords in the circle
        available = [(c, chord2idx[c]) for c in circle if c in chord2idx]
        
        if len(available) < 3:
            print(f"\n{name}: Not enough chords found")
            return None, []
        
        # Get embeddings
        chord_names = [c for c, _ in available]
        indices = [idx for _, idx in available]
        circle_embeddings = embeddings[indices]
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(circle_embeddings)
        
        # Calculate adjacent similarities (in circle order)
        adjacent_sims = []
        for i in range(len(chord_names)):
            j = (i + 1) % len(chord_names)
            sim = sim_matrix[i, j]
            adjacent_sims.append((chord_names[i], chord_names[j], sim))
        
        avg_adjacent_sim = np.mean([s for _, _, s in adjacent_sims])
        
        # Calculate opposite similarities (chords across the circle - tritone)
        opposite_sims = []
        for i in range(len(chord_names) // 2):
            j = (i + len(chord_names) // 2) % len(chord_names)
            if j < len(chord_names):
                sim = sim_matrix[i, j]
                opposite_sims.append((chord_names[i], chord_names[j], sim))
        
        avg_opposite_sim = np.mean([s for _, _, s in opposite_sims]) if opposite_sims else 0
        
        print(f"\n{name}:")
        print(f"  Chords found: {' → '.join(chord_names)}")
        print(f"  Adjacent pairs (fifths apart):")
        for c1, c2, sim in adjacent_sims:
            print(f"    {c1} → {c2}: {sim:.3f}")
        print(f"  Average adjacent similarity: {avg_adjacent_sim:.3f}")
        print(f"  Average tritone similarity: {avg_opposite_sim:.3f}")
        print(f"  Ratio (adj/tritone): {avg_adjacent_sim/avg_opposite_sim:.2f}x" if avg_opposite_sim > 0 else "")
        
        return sim_matrix, chord_names
    
    # Analyze major circle
    major_sim, major_chords = analyze_circle(circle_major, "Major Chords Circle")
    
    # Analyze minor circle
    minor_sim, minor_chords = analyze_circle(circle_minor, "Minor Chords Circle")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    def plot_circle(ax, sim_matrix, chord_names, title):
        """Plot a circle of fifths visualization."""
        if sim_matrix is None:
            ax.text(0.5, 0.5, "Not enough data", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return
        
        n = len(chord_names)
        
        # Draw circle
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        # Start from top (12 o'clock position) and go clockwise
        theta = np.pi/2 - theta
        
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Plot nodes
        ax.scatter(x, y, s=800, c='lightblue', edgecolors='navy', linewidths=2, zorder=3)
        
        # Add chord labels
        for i, chord in enumerate(chord_names):
            ax.annotate(chord, (x[i], y[i]), ha='center', va='center', 
                       fontsize=10, fontweight='bold', zorder=4)
        
        # Draw edges colored by similarity
        cmap = plt.cm.RdYlGn  # Red (low) to Green (high)
        
        for i in range(n):
            j = (i + 1) % n
            sim = sim_matrix[i, j]
            # Normalize similarity for color (typical range 0.3-0.8)
            norm_sim = np.clip((sim - 0.2) / 0.6, 0, 1)
            color = cmap(norm_sim)
            
            # Draw arc
            ax.annotate('', xy=(x[j], y[j]), xytext=(x[i], y[i]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2 + sim*3))
        
        # Draw tritone connections (dashed)
        for i in range(n // 2):
            j = (i + n // 2) % n
            if j < n:
                sim = sim_matrix[i, j]
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k--', alpha=0.2, lw=1)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.2, 0.8))
        sm.set_array([])
    
    plot_circle(axes[0], major_sim, major_chords, "Circle of Fifths (Major)")
    plot_circle(axes[1], minor_sim, minor_chords, "Circle of Fifths (Minor)")
    
    # Add colorbar
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0.2, 0.8))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Cosine Similarity (adjacent chords)', fontsize=10)
    
    plt.suptitle('Circle of Fifths Relationship in Chord Embeddings', fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved circle of fifths visualization to {output_path}")
    plt.close()
    
    # Additional analysis: Check if functional harmony is captured
    print("\n" + "-"*60)
    print("Functional Harmony Analysis:")
    print("-"*60)
    
    # I-IV-V-I cadence analysis
    cadence_chords = ['I', 'IV', 'V']
    if all(c in chord2idx for c in cadence_chords):
        indices = [chord2idx[c] for c in cadence_chords]
        cadence_emb = embeddings[indices]
        cadence_sim = cosine_similarity(cadence_emb)
        
        print(f"  Primary function chords (I, IV, V):")
        print(f"    I ↔ IV: {cadence_sim[0,1]:.3f}")
        print(f"    I ↔ V:  {cadence_sim[0,2]:.3f}")
        print(f"    IV ↔ V: {cadence_sim[1,2]:.3f}")
        print(f"    Average: {(cadence_sim[0,1] + cadence_sim[0,2] + cadence_sim[1,2])/3:.3f}")
    
    # Relative major/minor analysis
    rel_pairs = [('I', 'vim'), ('IV', 'iim'), ('V', 'iiim')]
    print(f"\n  Relative major/minor pairs:")
    for major, minor in rel_pairs:
        if major in chord2idx and minor in chord2idx:
            emb_maj = embeddings[chord2idx[major]]
            emb_min = embeddings[chord2idx[minor]]
            sim = cosine_similarity([emb_maj], [emb_min])[0, 0]
            print(f"    {major} ↔ {minor}: {sim:.3f}")
    
    return {'major_circle': major_chords, 'minor_circle': minor_chords}


def analyze_embedding_vs_cof_distance(
        embeddings: np.ndarray,
        idx2chord: Dict[int, str],
        output_path: str = 'embedding_vs_cof_distance.png') -> Dict:
    """
    Analyze the relationship between embedding cosine distance and
    circle-of-fifths (CoF) distance.

    For every unique pair of chords that appear in both the circle-of-fifths
    ordering and the trained vocabulary, this function computes:

        cosine distance  = 1 - cosine_similarity(emb_i, emb_j)
        CoF distance     = min(|pos_i - pos_j|, 12 - |pos_i - pos_j|)
                           (ranges 0–6, where 6 = tritone)

    It then produces a scatter plot of CoF distance vs cosine distance,
    overlays the per-step median, and reports the Spearman correlation.

    Args:
        embeddings: Chord embedding matrix [vocab_size, embedding_dim]
        idx2chord: Index to chord mapping
        output_path: Path to save the scatter plot

    Returns:
        Dictionary with Spearman rho, p-value, n_pairs, and per-step medians
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("analyze_embedding_vs_cof_distance requires matplotlib and sklearn")
        return {}

    chord2idx = {v: k for k, v in idx2chord.items()}

    # Circle of fifths in Roman numeral notation (12 positions)
    cof_major = ['I', 'V', 'II', 'VI', 'III', 'VII',
                 'bV', 'bII', 'bVI', 'bIII', 'bVII', 'IV']
    cof_minor = ['im', 'vm', 'iim', 'vim', 'iiim', 'viim',
                 'bvm', 'biim', 'bvim', 'biiim', 'bviim', 'ivm']

    def get_cof_pairs(circle, label):
        """Return (chord_i, chord_j, cof_dist, cos_dist) for all unique pairs."""
        available = [(c, chord2idx[c]) for c in circle if c in chord2idx]
        if len(available) < 2:
            print(f"  {label}: only {len(available)} chord(s) found in vocabulary, skipping.")
            return []

        n = len(circle)
        pos_map = {c: i for i, c in enumerate(circle)}

        pairs = []
        for k1 in range(len(available)):
            for k2 in range(k1 + 1, len(available)):
                c1, idx1 = available[k1]
                c2, idx2 = available[k2]

                # Circle-of-fifths distance (shortest arc length)
                diff = abs(pos_map[c1] - pos_map[c2])
                cof_dist = min(diff, n - diff)

                # Cosine distance = 1 - cosine similarity
                e1 = embeddings[idx1].astype(np.float64)
                e2 = embeddings[idx2].astype(np.float64)
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)
                if norm1 < 1e-10 or norm2 < 1e-10:
                    continue
                cos_sim = np.dot(e1, e2) / (norm1 * norm2)
                cos_dist = 1.0 - float(np.clip(cos_sim, -1.0, 1.0))

                pairs.append((c1, c2, cof_dist, cos_dist))
        return pairs

    major_pairs = get_cof_pairs(cof_major, "Major")
    minor_pairs = get_cof_pairs(cof_minor, "Minor")
    all_pairs = major_pairs + minor_pairs

    if not all_pairs:
        print("No chord pairs found. Skipping CoF distance analysis.")
        return {}

    cof_dists = np.array([p[2] for p in all_pairs], dtype=int)
    cos_dists = np.array([p[3] for p in all_pairs], dtype=float)

    # ── Spearman correlation ──────────────────────────────────────────
    print("\n" + "="*60)
    print("Embedding Distance vs Circle-of-Fifths Distance")
    print("="*60)
    print(f"  Total chord pairs analysed: {len(all_pairs)} "
          f"({len(major_pairs)} major + {len(minor_pairs)} minor)")

    spearman_result = {}
    rho, pval = None, None
    if HAS_SCIPY:
        rho, pval = spearmanr(cof_dists, cos_dists)
        print(f"  Spearman ρ = {rho:.4f}  (p = {pval:.4e})")
        spearman_result = {'rho': float(rho), 'p_value': float(pval)}
    else:
        print("  Install scipy for Spearman correlation (pip install scipy).")

    print("\n  Median cosine distance per CoF step:")
    for step in range(7):
        subset = cos_dists[cof_dists == step]
        if len(subset):
            print(f"    step {step}: n={len(subset):3d}  "
                  f"median={np.median(subset):.4f}  "
                  f"mean={subset.mean():.4f}  std={subset.std():.4f}")

    # ── Scatter plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    rng = np.random.default_rng(42)

    if major_pairs and minor_pairs:
        groups = [
            ('Major', major_pairs, '#2196F3', 'o'),
            ('Minor', minor_pairs, '#E91E63', '^'),
        ]
    else:
        groups = [('Chords', all_pairs, '#2196F3', 'o')]

    for label, pairs, color, marker in groups:
        cof_x = np.array([p[2] for p in pairs], dtype=float)
        cos_y = np.array([p[3] for p in pairs], dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=len(cof_x))
        ax.scatter(cof_x + jitter, cos_y,
                   alpha=0.55, s=35, c=color, marker=marker,
                   label=label, zorder=3)

    # Per-step median overlay
    steps_present = sorted(set(cof_dists.tolist()))
    medians = [np.median(cos_dists[cof_dists == s]) for s in steps_present]
    ax.plot(steps_present, medians, 'k-o', lw=2, markersize=7,
            zorder=5, label='Median per step')

    ax.set_xlabel('Circle-of-Fifths Distance (steps)', fontsize=13)
    ax.set_ylabel('Cosine Distance (1 − cosine similarity)', fontsize=13)
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(s) for s in range(7)])
    ax.grid(True, alpha=0.3)

    if rho is not None:
        title = (
            f'Embedding Cosine Distance vs Circle-of-Fifths Distance\n'
            f'Spearman \u03c1 = {rho:.4f},  p = {pval:.2e}  '
            f'(N\u202f=\u202f{len(all_pairs)})'
        )
    else:
        title = 'Embedding Cosine Distance vs Circle-of-Fifths Distance'
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved scatter plot to {output_path}")
    plt.close()

    per_step = {
        str(s): float(np.median(cos_dists[cof_dists == s]))
        for s in range(7)
        if (cof_dists == s).any()
    }
    return {'n_pairs': len(all_pairs), 'per_step_median': per_step, **spearman_result}


def main():
    parser = argparse.ArgumentParser(description='Chord2Vec visualization and evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: model_dir)')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE visualization')
    parser.add_argument('--similarity', action='store_true',
                        help='Analyze chord similarities')
    parser.add_argument('--analogy', action='store_true',
                        help='Evaluate chord analogies')
    parser.add_argument('--cluster', action='store_true',
                        help='Cluster chords')
    parser.add_argument('--history', action='store_true',
                        help='Plot training history')
    parser.add_argument('--circle_of_fifths', action='store_true',
                        help='Analyze circle of fifths relationships')
    parser.add_argument('--cof_distance', action='store_true',
                        help='Scatter plot: embedding cosine distance vs CoF distance + Spearman correlation')
    parser.add_argument('--ring_graph', action='store_true',
                        help='Draw multi-ring chord correlation graph with arrow-colored similarities')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    print(f"Loading model from {args.model_dir}...")
    model, chord2idx, idx2chord, embeddings = load_model_and_vocab(args.model_dir)
    print(f"Loaded model with {len(chord2idx)} chords")
    
    # Run requested analyses
    if args.all or args.tsne:
        visualize_embeddings_tsne(
            embeddings, idx2chord,
            output_path=os.path.join(output_dir, 'chord_embeddings_tsne.png'),
            highlight_chords=['I', 'IV', 'V', 'VI', 'II', 'III', 'VII']
        )
    
    if args.all or args.similarity:
        print("\nChord Similarity Analysis:")
        print("="*60)
        results = analyze_chord_similarities(model, chord2idx, idx2chord)
        for chord, similar in results.items():
            similar_str = ', '.join([f"{c}({s:.3f})" for c, s in similar[:5]])
            print(f"  {chord:10s} -> {similar_str}")
        
        # Also create similarity matrix
        compute_chord_similarity_matrix(
            embeddings, idx2chord,
            output_path=os.path.join(output_dir, 'similarity_matrix.png')
        )
    
    if args.all or args.analogy:
        print("\nChord Analogy Evaluation:")
        print("="*60)
        results = evaluate_analogies(model, chord2idx, idx2chord)
        print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
        for detail in results['details']:
            status = "✓" if detail['correct'] else "✗"
            print(f"  {status} {detail['analogy']} -> {detail['predicted']} (expected: {detail['expected']})")
    
    if args.all or args.cluster:
        cluster_chords(
            embeddings, idx2chord,
            output_path=os.path.join(output_dir, 'chord_clusters.json')
        )
    
    if args.all or args.history:
        history_path = os.path.join(args.model_dir, 'training_history.json')
        if os.path.exists(history_path):
            plot_training_history(
                history_path,
                output_path=os.path.join(output_dir, 'training_history.png')
            )
        else:
            print(f"Training history not found at {history_path}")
    
    if args.all or args.circle_of_fifths:
        analyze_circle_of_fifths(
            embeddings, idx2chord,
            output_path=os.path.join(output_dir, 'circle_of_fifths.png')
        )

    if args.all or args.cof_distance:
        analyze_embedding_vs_cof_distance(
            embeddings, idx2chord,
            output_path=os.path.join(output_dir, 'embedding_vs_cof_distance.png')
        )

    if args.all or args.ring_graph:
        plot_multiring_chord_correlation(
            embeddings,
            idx2chord,
            output_path=os.path.join(output_dir, 'multiring_chord_correlation.png'),
            edge_csv_path=os.path.join(output_dir, 'multiring_edge_correlations.csv'),
            edge_json_path=os.path.join(output_dir, 'multiring_edge_correlations.json'),
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
