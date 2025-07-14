#!/usr/bin/env python3

"""
Enhanced Confluence Document Embedder with Parallel Processing
Handles multiple batch JSON files with multiprocessing and memory optimization
"""

import json
import numpy as np
import pickle
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterator, Optional
from sentence_transformers import SentenceTransformer
import logging
import argparse
import glob
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project structure paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEXES_DIR = DATA_DIR / "indexes"
LOGS_DIR = PROJECT_ROOT / "logs"

class DocumentChunker:
    """Intelligent document chunking for better search results"""
    
    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with smart boundaries"""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                sentence_end = text.rfind('. ', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('\n', start, end)
                if sentence_end != -1 and sentence_end > start + self.max_chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_documents(self, docs: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Chunk documents and return texts with comprehensive metadata"""
        texts = []
        metadata = []
        
        for doc in docs:
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            
            # Create title + content combination
            full_text = f"{title}\n\n{content}"
            chunks = self.chunk_text(full_text)
            
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadata.append({
                    'doc_id': doc.get('id', 'unknown'),
                    'title': title,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'attachments': doc.get('attachments', []),
                    'space': doc.get('space', ''),
                    'version': doc.get('version', ''),
                    'creator': doc.get('creator', ''),
                    'creationDate': doc.get('creationDate', ''),
                    'lastModificationDate': doc.get('lastModificationDate', ''),
                    'word_count': doc.get('word_count', 0),
                    'chunk_text': chunk[:200] + '...' if len(chunk) > 200 else chunk,
                    'chunk_length': len(chunk)
                })
        
        return texts, metadata

def process_batch_file(args_tuple: Tuple[str, str, int, int, int]) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    """Process a single batch file - designed for multiprocessing"""
    batch_file, model_name, chunk_size, overlap, batch_idx = args_tuple
    
    try:
        # Initialize model in each process
        model = SentenceTransformer(model_name)
        chunker = DocumentChunker(max_chunk_size=chunk_size, overlap=overlap)
        
        # Load and process documents
        with open(batch_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        # Chunk documents
        texts, metadata = chunker.chunk_documents(docs)
        
        if not texts:
            logger.warning(f"No texts found in batch {batch_idx}: {batch_file}")
            return np.array([]), [], batch_idx
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Clean up
        del model
        gc.collect()
        
        logger.info(f"Processed batch {batch_idx}: {len(docs)} docs → {len(texts)} chunks → {embeddings.shape}")
        
        return embeddings, metadata, batch_idx
        
    except Exception as e:
        logger.error(f"Failed to process batch {batch_idx} ({batch_file}): {e}")
        return np.array([]), [], batch_idx

class ParallelConfluenceEmbedder:
    """Enhanced embedder with parallel processing for large datasets"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 512,
                 overlap: int = 50,
                 n_processes: Optional[int] = None):
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_processes = n_processes or min(mp.cpu_count(), 8)  # Limit to 8 processes max
        
        # Create directories
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized embedder with model: {model_name}")
        logger.info(f"Using {self.n_processes} parallel processes")
        
        # Test model loading
        logger.info("Testing model loading...")
        test_model = SentenceTransformer(model_name)
        self.embedding_dim = test_model.get_sentence_embedding_dimension()
        del test_model
        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def find_batch_files(self, batch_dir: str, pattern: str = "confluence_batch_*.json") -> List[str]:
        """Find and sort batch files"""
        batch_pattern = Path(batch_dir) / pattern
        all_files = sorted(glob.glob(str(batch_pattern)))
        
        # Filter out manifest files
        batch_files = [f for f in all_files if 'manifest' not in Path(f).name.lower()]
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found matching pattern: {batch_pattern}")
        
        logger.info(f"Found {len(batch_files)} batch files")
        return batch_files
    
    def embed_batch_files_parallel(self, 
                                 batch_files: List[str], 
                                 output_prefix: str = "confluence") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process multiple batch files in parallel"""
        
        logger.info(f"Starting parallel embedding of {len(batch_files)} batch files")
        logger.info(f"Using {self.n_processes} processes")
        
        # Prepare arguments for parallel processing
        process_args = [
            (batch_file, self.model_name, self.chunk_size, self.overlap, idx)
            for idx, batch_file in enumerate(batch_files)
        ]
        
        # Process batches in parallel
        all_embeddings = []
        all_metadata = []
        completed_batches = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit all jobs
            future_to_batch = {
                executor.submit(process_batch_file, args): args[4] 
                for args in process_args
            }
            
            # Process completed jobs
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    embeddings, metadata, returned_idx = future.result()
                    
                    if len(embeddings) > 0:
                        all_embeddings.append(embeddings)
                        all_metadata.extend(metadata)
                    
                    completed_batches += 1
                    elapsed = time.time() - start_time
                    
                    logger.info(f"Completed batch {completed_batches}/{len(batch_files)} "
                              f"(batch_idx: {batch_idx}) in {elapsed:.1f}s")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    continue
        
        # Combine all embeddings
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.array([]).reshape(0, self.embedding_dim)
        
        total_time = time.time() - start_time
        logger.info(f"Parallel processing complete in {total_time:.1f}s")
        logger.info(f"Generated {len(final_embeddings)} total embeddings from {len(all_metadata)} chunks")
        
        return final_embeddings, all_metadata
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       metadata: List[Dict[str, Any]], 
                       output_prefix: str = "confluence",
                       job_id: Optional[str] = None) -> Dict[str, Any]:
        """Save embeddings and metadata with comprehensive summary"""
        
        # Generate job ID if not provided
        if not job_id:
            job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filenames
        embeddings_file = EMBEDDINGS_DIR / f'{output_prefix}_embeddings_{job_id}.npy'
        metadata_file = EMBEDDINGS_DIR / f'{output_prefix}_metadata_{job_id}.pkl'
        summary_file = EMBEDDINGS_DIR / f'{output_prefix}_summary_{job_id}.json'
        
        # Save embeddings
        np.save(embeddings_file, embeddings)
        logger.info(f"Embeddings saved: {embeddings_file}")
        
        # Save metadata
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved: {metadata_file}")
        
        # Create comprehensive summary
        unique_docs = len(set(m['doc_id'] for m in metadata))
        spaces = set(m['space'] for m in metadata if m['space'])
        avg_chunk_length = np.mean([m['chunk_length'] for m in metadata])
        
        summary = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else self.embedding_dim,
            'total_embeddings': len(embeddings),
            'total_chunks': len(metadata),
            'unique_documents': unique_docs,
            'unique_spaces': len(spaces),
            'spaces': sorted(list(spaces)),
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'avg_chunk_length': float(avg_chunk_length),
            'n_processes': self.n_processes,
            'files': {
                'embeddings': str(embeddings_file),
                'metadata': str(metadata_file),
                'summary': str(summary_file)
            },
            'statistics': {
                'embeddings_shape': embeddings.shape if len(embeddings) > 0 else [0, self.embedding_dim],
                'embeddings_dtype': str(embeddings.dtype) if len(embeddings) > 0 else 'float32',
                'embeddings_size_mb': embeddings.nbytes / (1024 * 1024) if len(embeddings) > 0 else 0
            }
        }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {summary_file}")
        
        return summary
    
    def process_confluence_batches(self, 
                                 batch_dir: str = None,
                                 batch_pattern: str = "confluence_batch_*.json",
                                 output_prefix: str = "confluence",
                                 job_id: Optional[str] = None) -> Dict[str, Any]:
        """Complete pipeline: find batches, embed, and save"""
        
        # Use default batch directory if not provided
        if batch_dir is None:
            batch_dir = str(PARSED_DIR)
        
        # Find batch files
        batch_files = self.find_batch_files(batch_dir, batch_pattern)
        
        # Process in parallel
        embeddings, metadata = self.embed_batch_files_parallel(batch_files, output_prefix)
        
        # Save results
        summary = self.save_embeddings(embeddings, metadata, output_prefix, job_id)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Enhanced Confluence Embedder with Parallel Processing')
    
    # Input options
    parser.add_argument('--batch-dir', default=str(PARSED_DIR),
                       help='Directory containing batch JSON files')
    parser.add_argument('--batch-pattern', default='confluence_batch_*.json',
                       help='Pattern to match batch files')
    parser.add_argument('--batch-files', nargs='+',
                       help='Specific batch files to process (overrides batch-dir)')
    
    # Model and processing options
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Maximum chunk size in characters')
    parser.add_argument('--overlap', type=int, default=50,
                       help='Overlap between chunks in characters')
    parser.add_argument('--processes', type=int,
                       help='Number of parallel processes (default: auto)')
    
    # Output options
    parser.add_argument('--output-prefix', default='confluence',
                       help='Prefix for output files')
    parser.add_argument('--job-id',
                       help='Job ID for output files (default: timestamp)')
    
    # Utility options
    parser.add_argument('--test-model', action='store_true',
                       help='Test model loading and exit')
    parser.add_argument('--list-batches', action='store_true',
                       help='List found batch files and exit')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize embedder
        embedder = ParallelConfluenceEmbedder(
            model_name=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            n_processes=args.processes
        )
        
        # Test model loading if requested
        if args.test_model:
            print(f"✅ Model '{args.model}' loaded successfully")
            print(f"📐 Embedding dimension: {embedder.embedding_dim}")
            return 0
        
        # List batches if requested
        if args.list_batches:
            batch_files = embedder.find_batch_files(args.batch_dir, args.batch_pattern)
            print(f"Found {len(batch_files)} batch files:")
            for i, batch_file in enumerate(batch_files, 1):
                print(f"  {i:3d}: {batch_file}")
            return 0
        
        # Process specific batch files or discover them
        if args.batch_files:
            logger.info(f"Processing {len(args.batch_files)} specified batch files")
            embeddings, metadata = embedder.embed_batch_files_parallel(args.batch_files, args.output_prefix)
            summary = embedder.save_embeddings(embeddings, metadata, args.output_prefix, args.job_id)
        else:
            # Full pipeline
            summary = embedder.process_confluence_batches(
                batch_dir=args.batch_dir,
                batch_pattern=args.batch_pattern,
                output_prefix=args.output_prefix,
                job_id=args.job_id
            )
        
        # Print results
        print(f"\n✅ Embedding generation complete!")
        print(f"🆔 Job ID: {summary['job_id']}")
        print(f"📊 Generated {summary['total_embeddings']:,} embeddings from {summary['total_chunks']:,} chunks")
        print(f"📄 Processed {summary['unique_documents']:,} unique documents")
        print(f"🏢 Found {summary['unique_spaces']} unique spaces")
        print(f"🔧 Model: {summary['model_name']}")
        print(f"📐 Embedding dimension: {summary['embedding_dimension']}")
        print(f"💾 Embeddings size: {summary['statistics']['embeddings_size_mb']:.1f} MB")
        print(f"📁 Files saved to: {EMBEDDINGS_DIR}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return 1

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    exit(main())
