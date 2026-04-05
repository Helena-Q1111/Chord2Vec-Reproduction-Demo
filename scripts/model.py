"""
Chord2Vec Model Implementation in PyTorch.

This module implements the chord2vec model using the skip-gram architecture,
similar to word2vec but adapted for chord sequences.

Three model variants are provided:
1. SkipGramModel - Basic skip-gram with negative sampling
2. LinearChord2Vec - Linear model assuming note independence
3. Seq2SeqChord2Vec - Sequence-to-sequence model with LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class ChordDataset(Dataset):
    """Dataset for chord2vec skip-gram training."""
    
    def __init__(self, pairs: List[Tuple[int, int]]):
        """
        Args:
            pairs: List of (center_chord_idx, context_chord_idx) pairs
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class NegativeSampler:
    """
    Negative sampling for skip-gram training.
    Uses word frequency raised to 3/4 power as in word2vec.
    """
    
    def __init__(self, vocab_size: int, chord_counts: Optional[Dict[int, int]] = None):
        """
        Args:
            vocab_size: Size of chord vocabulary
            chord_counts: Dictionary mapping chord index to count
        """
        self.vocab_size = vocab_size
        
        if chord_counts:
            # Create sampling distribution
            counts = np.array([chord_counts.get(i, 1) for i in range(vocab_size)])
            # Raise to 3/4 power as in word2vec
            weights = np.power(counts, 0.75)
            self.probs = weights / weights.sum()
        else:
            # Uniform distribution
            self.probs = np.ones(vocab_size) / vocab_size
    
    def sample(self, n_samples: int, exclude: Optional[List[int]] = None) -> np.ndarray:
        """Sample negative examples."""
        samples = np.random.choice(self.vocab_size, size=n_samples, p=self.probs)
        
        # Optionally exclude certain indices (like the positive examples)
        if exclude:
            exclude_set = set(exclude)
            mask = np.array([s not in exclude_set for s in samples])
            while not mask.all():
                n_resample = (~mask).sum()
                new_samples = np.random.choice(self.vocab_size, size=n_resample, p=self.probs)
                samples[~mask] = new_samples
                mask = np.array([s not in exclude_set for s in samples])
        
        return samples


class SkipGramChord2Vec(nn.Module):
    """
    Skip-gram model for chord embeddings with negative sampling.
    
    The model learns to predict context chords given a center chord,
    similar to word2vec skip-gram.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            vocab_size: Number of unique chords in vocabulary
            embedding_dim: Dimension of chord embeddings
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Center (input) embeddings
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context (output) embeddings
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier initialization for better convergence."""
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center_idx: torch.Tensor, 
                context_idx: torch.Tensor,
                negative_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with negative sampling.
        
        Args:
            center_idx: Center chord indices [batch_size]
            context_idx: Positive context chord indices [batch_size]
            negative_idx: Negative sample indices [batch_size, n_negative]
        
        Returns:
            pos_score: Score for positive pairs
            neg_score: Score for negative pairs
        """
        # Get embeddings with dropout for regularization
        center_emb = self.dropout(self.center_embeddings(center_idx))  # [batch, emb_dim]
        context_emb = self.dropout(self.context_embeddings(context_idx))  # [batch, emb_dim]
        neg_emb = self.dropout(self.context_embeddings(negative_idx))  # [batch, n_neg, emb_dim]
        
        # Positive score: dot product between center and context
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [batch]
        pos_score = F.logsigmoid(pos_score)
        
        # Negative score: dot product between center and negatives
        # center_emb: [batch, emb_dim] -> [batch, emb_dim, 1]
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # [batch, n_neg]
        neg_score = F.logsigmoid(-neg_score).sum(dim=1)  # [batch]
        
        return pos_score, neg_score
    
    def get_embedding(self, chord_idx: torch.Tensor) -> torch.Tensor:
        """Get the embedding for chord indices."""
        return self.center_embeddings(chord_idx)
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all chord embeddings as numpy array."""
        return self.center_embeddings.weight.data.cpu().numpy()
    
    def most_similar(self, chord_idx: int, idx2chord: Dict[int, str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar chords to a given chord.
        
        Args:
            chord_idx: Index of query chord
            idx2chord: Mapping from index to chord symbol
            top_k: Number of similar chords to return
        
        Returns:
            List of (chord_symbol, similarity_score) tuples
        """
        with torch.no_grad():
            query_emb = self.center_embeddings.weight[chord_idx]
            all_emb = self.center_embeddings.weight
            
            # Compute cosine similarity
            similarities = F.cosine_similarity(query_emb.unsqueeze(0), all_emb, dim=1)
            
            # Get top-k (excluding self)
            top_vals, top_idx = torch.topk(similarities, k=top_k + 1)
            
            results = []
            for val, idx in zip(top_vals.cpu().numpy(), top_idx.cpu().numpy()):
                if idx != chord_idx:
                    chord_name = idx2chord.get(int(idx), f"UNK_{idx}")
                    results.append((chord_name, float(val)))
            
            return results[:top_k]


class LinearChord2Vec(nn.Module):
    """
    Linear chord2vec model.
    
    Assumes conditional independence between notes in context chord
    given the center chord. Uses binary cross-entropy for multi-label prediction.
    
    This is equivalent to the linear model in the original chord2vec paper.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256):
        """
        Args:
            vocab_size: Number of unique chords
            embedding_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding layer (input chord -> hidden)
        self.encoder = nn.Linear(vocab_size, embedding_dim, bias=False)
        
        # Output layer (hidden -> output chord distribution)
        self.decoder = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, center_onehot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            center_onehot: One-hot encoded center chord [batch, vocab_size]
        
        Returns:
            Output logits for context chord prediction [batch, vocab_size]
        """
        # Normalize input (in case chord has multiple notes)
        center_normalized = center_onehot / (center_onehot.sum(dim=1, keepdim=True).clamp(min=1))
        
        # Encode
        hidden = self.encoder(center_normalized)
        
        # Decode
        output = self.decoder(hidden)
        
        return output
    
    def get_embedding(self, chord_onehot: torch.Tensor) -> torch.Tensor:
        """Get embedding for a chord."""
        with torch.no_grad():
            normalized = chord_onehot / (chord_onehot.sum(dim=1, keepdim=True).clamp(min=1))
            return self.encoder(normalized)


class Seq2SeqChord2Vec(nn.Module):
    """
    Sequence-to-sequence chord2vec model using LSTM.
    
    Uses an encoder-decoder architecture to learn chord representations
    that are useful for predicting sequences of context chords.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            vocab_size: Number of unique chords
            embedding_dim: Dimension of chord embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, input_seq: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode input chord sequence.
        
        Args:
            input_seq: Input chord indices [batch, seq_len]
        
        Returns:
            encoder_output: Encoder outputs [batch, seq_len, hidden]
            hidden: Final hidden state
        """
        embedded = self.dropout(self.embedding(input_seq))
        output, hidden = self.encoder(embedded)
        return output, hidden
    
    def decode(self, target_seq: torch.Tensor, hidden: Tuple) -> torch.Tensor:
        """
        Decode to predict context chord sequence.
        
        Args:
            target_seq: Target chord indices [batch, seq_len]
            hidden: Initial hidden state from encoder
        
        Returns:
            Output logits [batch, seq_len, vocab_size]
        """
        embedded = self.dropout(self.embedding(target_seq))
        output, _ = self.decoder(embedded, hidden)
        logits = self.output_proj(output)
        return logits
    
    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_seq: Input chord sequence [batch, input_len]
            target_seq: Target context sequence [batch, target_len]
        
        Returns:
            Output logits [batch, target_len, vocab_size]
        """
        _, hidden = self.encode(input_seq)
        logits = self.decode(target_seq, hidden)
        return logits
    
    def get_embedding(self, chord_seq: torch.Tensor) -> torch.Tensor:
        """
        Get the fixed-length representation of a chord sequence.
        Uses the final encoder hidden state.
        """
        with torch.no_grad():
            _, (hidden, _) = self.encode(chord_seq)
            # Use last layer hidden state
            return hidden[-1]


class Chord2VecTrainer:
    """
    Trainer class for chord2vec models.
    """
    
    def __init__(self, model: nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_negative: int = 5,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 grad_clip: float = 1.0):
        """
        Args:
            model: Chord2vec model
            optimizer: PyTorch optimizer
            device: Device to train on
            n_negative: Number of negative samples
            scheduler: Learning rate scheduler (optional)
            grad_clip: Gradient clipping max norm (0 to disable)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.n_negative = n_negative
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        
        # Initialize negative sampler
        self.neg_sampler = NegativeSampler(model.vocab_size)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for center, context in dataloader:
            center = center.to(self.device)
            context = context.to(self.device)
            
            # Sample negatives
            batch_size = center.size(0)
            negative = self.neg_sampler.sample(batch_size * self.n_negative)
            negative = torch.tensor(negative, dtype=torch.long).view(batch_size, self.n_negative)
            negative = negative.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pos_score, neg_score = self.model(center, context, negative)
            
            # Loss is negative of sum of positive and negative scores
            loss = -(pos_score + neg_score).mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Step scheduler per batch for per-step schedulers (OneCycleLR, LambdaLR with warmup)
            if self.scheduler is not None:
                # Step all schedulers except ReduceLROnPlateau per batch
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for center, context in dataloader:
                center = center.to(self.device)
                context = context.to(self.device)
                
                # Sample negatives
                batch_size = center.size(0)
                negative = self.neg_sampler.sample(batch_size * self.n_negative)
                negative = torch.tensor(negative, dtype=torch.long).view(batch_size, self.n_negative)
                negative = negative.to(self.device)
                
                # Forward pass
                pos_score, neg_score = self.model(center, context, negative)
                loss = -(pos_score + neg_score).mean()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches


def compute_analogy(model: SkipGramChord2Vec, 
                    idx2chord: Dict[int, str],
                    chord2idx: Dict[str, int],
                    a: str, b: str, c: str, 
                    top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Compute chord analogy: a is to b as c is to ?
    
    Example: I is to V as IV is to ? (might return I or similar)
    
    Args:
        model: Trained chord2vec model
        idx2chord: Index to chord mapping
        chord2idx: Chord to index mapping
        a, b, c: Chord symbols for analogy
        top_k: Number of results to return
    
    Returns:
        List of (chord_symbol, score) tuples
    """
    if a not in chord2idx or b not in chord2idx or c not in chord2idx:
        return []
    
    with torch.no_grad():
        emb_a = model.center_embeddings.weight[chord2idx[a]]
        emb_b = model.center_embeddings.weight[chord2idx[b]]
        emb_c = model.center_embeddings.weight[chord2idx[c]]
        
        # Compute: b - a + c
        query = emb_b - emb_a + emb_c
        
        # Find most similar
        all_emb = model.center_embeddings.weight
        similarities = F.cosine_similarity(query.unsqueeze(0), all_emb, dim=1)
        
        # Exclude a, b, c from results
        exclude_idx = [chord2idx[a], chord2idx[b], chord2idx[c]]
        for idx in exclude_idx:
            similarities[idx] = -float('inf')
        
        top_vals, top_indices = torch.topk(similarities, k=top_k)
        
        results = []
        for val, idx in zip(top_vals.cpu().numpy(), top_indices.cpu().numpy()):
            chord_name = idx2chord.get(int(idx), f"UNK_{idx}")
            results.append((chord_name, float(val)))
        
        return results
