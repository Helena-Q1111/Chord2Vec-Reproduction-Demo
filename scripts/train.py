"""
Training script for chord2vec models.

Usage:
    python train.py --data Hooktheory.json.gz --epochs 50 --embedding_dim 128
"""

import argparse
import os
import json
import time
import pickle
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_processing import (
    prepare_dataset, 
    load_processed_data, 
    load_hooktheory_data, 
    audit_hooktheory_chord_coverage,
    extract_chord_sequences, 
    build_vocabulary, 
    generate_skipgram_pairs, 
    split_data
)
from model import SkipGramChord2Vec, LinearChord2Vec, ChordDataset, Chord2VecTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train chord2vec model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='Hooktheory.json.gz',
                        help='Path to Hooktheory.json.gz')
    parser.add_argument('--processed_data', type=str, default=None,
                        help='Path to preprocessed data (skip preprocessing)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--notation', type=str, default='roman', choices=['roman', 'absolute'],
                        help='Chord symbol notation style')
    parser.add_argument('--no_annotate_unknown', action='store_true',
                        help='Skip unknown interval patterns instead of tagging as unk[...]')
    parser.add_argument('--auto_quality_min_count', type=int, default=4,
                        help='Auto-recognize unknown interval patterns occurring at least this many times')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='skipgram',
                        choices=['skipgram', 'linear'],
                        help='Model type')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension (increased from 128)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Context window size (increased from 2)')
    parser.add_argument('--min_count', type=int, default=2,
                        help='Minimum chord count for vocabulary')
    parser.add_argument('--subsample', type=float, default=1e-4,
                        help='Subsampling threshold for frequent chords (0 to disable)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (increased from 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Peak learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--n_negative', type=int, default=15,
                        help='Number of negative samples (increased from 5)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience (increased from 5)')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'plateau', 'onecycle'],
                        help='Learning rate scheduler')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data_from_hooktheory(args):
    """Prepare training data from Hooktheory dataset."""
    print("\n" + "="*60)
    print("Preparing data from Hooktheory dataset...")
    print("="*60)
    
    # Load raw Hooktheory data
    data = load_hooktheory_data(args.data)

    # Coverage audit: check if all harmony interval patterns are recognized.
    coverage_report = audit_hooktheory_chord_coverage(
        data,
        auto_quality_min_count=args.auto_quality_min_count,
    )
    print("\nChord interval coverage audit:")
    print(f"  total_harmony_events:   {coverage_report['total_harmony_events']}")
    print(f"  known_harmony_events:   {coverage_report['known_harmony_events']} ({coverage_report['known_coverage']:.2%})")
    print(f"  unknown_harmony_events: {coverage_report['unknown_harmony_events']} ({coverage_report['unknown_coverage']:.2%})")
    print(f"  known_after_auto:       {coverage_report['known_harmony_events_after_auto']} ({coverage_report['known_coverage_after_auto']:.2%})")
    print(f"  unknown_after_auto:     {coverage_report['unknown_harmony_events_after_auto']} ({coverage_report['unknown_coverage_after_auto']:.2%})")

    audit_path = os.path.join(args.output_dir, 'preprocessing_coverage_audit.json')
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(coverage_report, f, indent=2, ensure_ascii=False)
    print(f"Saved preprocessing audit to {audit_path}")
    
    # Inspect data structure
    sample_key = list(data.keys())[0]
    sample = data[sample_key]
    print(f"\nSample data structure for '{sample_key}':")
    print(f"  Keys: {list(sample.keys())}")
    if 'annotations' in sample:
        print(f"  Annotations keys: {list(sample['annotations'].keys())}")
    
    # Extract chord sequences using the correct Hooktheory format
    chord_sequences = extract_chord_sequences(
        data,
        filter_audio_available=False,
        filter_melody=False,
        filter_harmony=True,
        exclude_tempo_changes=False,
        use_roman=(args.notation == 'roman'),
        annotate_unknown=(not args.no_annotate_unknown),
        auto_quality_min_count=args.auto_quality_min_count,
    )
    
    if not chord_sequences:
        print("ERROR: Could not extract chord sequences from data")
        return None
    
    # Build vocabulary
    chord2idx, idx2chord = build_vocabulary(chord_sequences, args.min_count)
    
    # Split data
    train_seqs, valid_seqs, test_seqs = split_data(chord_sequences)
    
    # Generate skip-gram pairs (with subsampling for training set only)
    print(f"\nGenerating skip-gram pairs with window_size={args.window_size}, subsample={args.subsample}")
    train_pairs = generate_skipgram_pairs(train_seqs, chord2idx, args.window_size, subsample_threshold=args.subsample)
    valid_pairs = generate_skipgram_pairs(valid_seqs, chord2idx, args.window_size, subsample_threshold=0)  # No subsampling for validation
    test_pairs = generate_skipgram_pairs(test_seqs, chord2idx, args.window_size, subsample_threshold=0)  # No subsampling for test
    
    return {
        'chord2idx': chord2idx,
        'idx2chord': idx2chord,
        'vocab_size': len(chord2idx),
        'train_pairs': train_pairs,
        'valid_pairs': valid_pairs,
        'test_pairs': test_pairs,
        'train_sequences': train_seqs,
        'valid_sequences': valid_seqs,
        'test_sequences': test_seqs,
    }


def train(args):
    """Main training function."""
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nUsing device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or prepare data
    if args.processed_data and os.path.exists(args.processed_data):
        print(f"\nLoading preprocessed data from {args.processed_data}")
        data = load_processed_data(args.processed_data)
    else:
        data = prepare_data_from_hooktheory(args)
        
        if data is None:
            print("Failed to prepare data. Exiting.")
            return
        
        # Save processed data
        processed_path = os.path.join(args.output_dir, 'processed_data.pkl')
        with open(processed_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {processed_path}")
    
    # Create datasets
    train_dataset = ChordDataset(data['train_pairs'])
    valid_dataset = ChordDataset(data['valid_pairs'])
    test_dataset = ChordDataset(data['test_pairs'])
    
    print(f"\nDataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training data. Check data preprocessing.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    vocab_size = data['vocab_size']
    print(f"\nVocabulary size: {vocab_size}")
    
    if args.model_type == 'skipgram':
        model = SkipGramChord2Vec(vocab_size, args.embedding_dim, dropout=args.dropout)
    elif args.model_type == 'linear':
        model = LinearChord2Vec(vocab_size, args.embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model: {args.model_type} with embedding_dim={args.embedding_dim}, dropout={args.dropout}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create learning rate scheduler
    scheduler = None
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    
    if args.scheduler == 'cosine':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"Using Cosine scheduler with {args.warmup_epochs} warmup epochs")
        
    elif args.scheduler == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=args.min_lr)
        print(f"Using ReduceLROnPlateau scheduler")
        
    elif args.scheduler == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        print(f"Using OneCycleLR scheduler")
    
    # Create trainer with scheduler and gradient clipping
    trainer = Chord2VecTrainer(
        model, optimizer, device, args.n_negative,
        scheduler=scheduler if args.scheduler not in ['plateau', 'none'] else None,
        grad_clip=args.grad_clip
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_valid_loss = float('inf')
    patience_counter = 0
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        valid_loss = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        
        epoch_time = time.time() - start_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Step plateau scheduler with validation loss
        if args.scheduler == 'plateau' and scheduler is not None:
            scheduler.step(valid_loss)
        
        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'vocab_size': vocab_size,
                'embedding_dim': args.embedding_dim,
                'model_type': args.model_type,
            }, model_path)
            print(f"  -> Saved best model (valid_loss: {valid_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.early_stopping} epochs)")
                break
    
    # Load best model for evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss = trainer.evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Save embeddings
    embeddings = model.get_all_embeddings()
    embedding_path = os.path.join(args.output_dir, 'chord_embeddings.npy')
    import numpy as np
    np.save(embedding_path, embeddings)
    print(f"Saved embeddings to {embedding_path}")
    
    # Save vocabulary with embeddings
    vocab_path = os.path.join(args.output_dir, 'vocabulary.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'chord2idx': data['chord2idx'],
            'idx2chord': {str(k): v for k, v in data['idx2chord'].items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'test_loss': test_loss,
            'best_epoch': checkpoint['epoch'],
            'args': vars(args),
        }, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    # Demo: find similar chords
    if args.model_type == 'skipgram':
        print("\n" + "="*60)
        print("Demo: Most similar chords")
        print("="*60)
        
        # Find most similar chords for common chords
        test_chords = ['I', 'IV', 'V', 'VI', 'II', 'III', 'VII']
        for chord in test_chords:
            if chord in data['chord2idx']:
                idx = data['chord2idx'][chord]
                similar = model.most_similar(idx, data['idx2chord'], top_k=5)
                similar_str = ', '.join([f"{c}({s:.3f})" for c, s in similar])
                print(f"  {chord:8s} -> {similar_str}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Output saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    args = parse_args()
    train(args)
