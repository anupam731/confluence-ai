#!/usr/bin/env python3

"""
Enhanced Confluence FAISS Indexer with Auto-Discovery and HNSW Optimization
Handles multiple embedding files with intelligent index selection for scalability
"""

import numpy as np
import faiss
import pickle
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse
from datetime import datetime
import time
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

class EnhancedConfluenceIndexer:
    """Enhanced FAISS indexer with auto-discovery and intelligent index selection"""
    
    def __init__(self, index_dir: str = None):
        self.index_dir = Path(index_dir) if index_dir else INDEXES_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Indexer initialized with output directory: {self.index_dir}")
    
    def discover_embedding_files(self, embeddings_dir: str = None) -> List[Tuple[str, str, str]]:
        """Intelligently discover and validate embedding file pairs"""
        if embeddings_dir is None:
            embeddings_dir = str(EMBEDDINGS_DIR)
        
        embeddings_dir = Path(embeddings_dir)
        embedding_files = list(embeddings_dir.glob("*_embeddings_*.npy"))
        
        file_pairs = []
        validation_summary = {
            'total_found': len(embedding_files),
            'valid_pairs': 0,
            'missing_metadata': 0,
            'empty_files': 0,
            'corrupted_files': 0
        }
        
        for emb_file in embedding_files:
            filename = emb_file.stem
            if "_embeddings_" in filename:
                job_id = filename.split("_embeddings_")[-1]
                prefix = filename.split("_embeddings_")[0]
                
                metadata_file = embeddings_dir / f"{prefix}_metadata_{job_id}.pkl"
                summary_file = embeddings_dir / f"{prefix}_summary_{job_id}.json"
                
                # INTELLIGENT VALIDATION
                try:
                    # Check if files exist
                    if not metadata_file.exists():
                        logger.warning(f"⚠️  Missing metadata for {emb_file.name}")
                        validation_summary['missing_metadata'] += 1
                        continue
                    
                    # Quick validation of file contents
                    embeddings = np.load(emb_file)
                    if embeddings.size == 0:
                        logger.warning(f"⚠️  Empty embeddings file: {emb_file.name}")
                        validation_summary['empty_files'] += 1
                        continue
                    
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                        if len(metadata) == 0:
                            logger.warning(f"⚠️  Empty metadata file: {metadata_file.name}")
                            validation_summary['empty_files'] += 1
                            continue
                    
                    # File pair is valid
                    file_pairs.append((str(emb_file), str(metadata_file), str(summary_file)))
                    validation_summary['valid_pairs'] += 1
                    logger.info(f"✅ Validated pair: {emb_file.name} + {metadata_file.name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Corrupted files for {emb_file.name}: {e}")
                    validation_summary['corrupted_files'] += 1
                    continue
        
        # INTELLIGENT SUMMARY
        logger.info(f"🔍 File discovery summary:")
        logger.info(f"   📁 Total embedding files found: {validation_summary['total_found']}")
        logger.info(f"   ✅ Valid pairs: {validation_summary['valid_pairs']}")
        logger.info(f"   ⚠️  Missing metadata: {validation_summary['missing_metadata']}")
        logger.info(f"   📭 Empty files: {validation_summary['empty_files']}")
        logger.info(f"   💥 Corrupted files: {validation_summary['corrupted_files']}")
        
        if not file_pairs:
            raise FileNotFoundError(f"❌ No valid embedding file pairs found in {embeddings_dir}. "
                                  f"Check your embedding generation process.")
        
        return file_pairs
    
    def load_embedding_data(self, embedding_file: str, metadata_file: str, summary_file: str = None) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Load embeddings, metadata, and summary data"""
        logger.info(f"Loading embeddings from {embedding_file}")
        embeddings = np.load(embedding_file)
        
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        summary = {}
        if summary_file and Path(summary_file).exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embeddings and {len(metadata)} metadata entries")
        return embeddings, metadata, summary
    
    def combine_multiple_embeddings(self, file_pairs: List[Tuple[str, str, str]]) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Intelligently combine multiple embedding files, automatically handling empty files"""
        logger.info(f"Intelligently processing {len(file_pairs)} embedding files")
        
        valid_embeddings = []
        all_metadata = []
        skipped_files = []
        combined_summary = {
            'source_files': [],
            'skipped_files': [],
            'total_embeddings': 0,
            'total_chunks': 0,
            'unique_documents': set(),
            'unique_spaces': set(),
            'combined_timestamp': datetime.now().isoformat()
        }
        
        for i, (emb_file, meta_file, summary_file) in enumerate(file_pairs):
            logger.info(f"Processing file pair {i+1}/{len(file_pairs)}: {Path(emb_file).name}")
            
            try:
                embeddings, metadata, summary = self.load_embedding_data(emb_file, meta_file, summary_file)
                
                # INTELLIGENT EMPTY FILE DETECTION
                if len(embeddings) == 0 or len(metadata) == 0:
                    logger.warning(f"⚠️  Skipping empty file: {Path(emb_file).name} (embeddings: {len(embeddings)}, metadata: {len(metadata)})")
                    skipped_files.append({
                        'file': emb_file,
                        'reason': 'empty_embeddings' if len(embeddings) == 0 else 'empty_metadata',
                        'embeddings_count': len(embeddings),
                        'metadata_count': len(metadata)
                    })
                    combined_summary['skipped_files'].append({
                        'embeddings': emb_file,
                        'metadata': meta_file,
                        'reason': 'empty_file',
                        'embeddings_count': len(embeddings),
                        'metadata_count': len(metadata)
                    })
                    continue
                
                # INTELLIGENT DIMENSION VALIDATION
                if valid_embeddings and embeddings.shape[1] != valid_embeddings[0].shape[1]:
                    logger.warning(f"⚠️  Skipping file with mismatched dimensions: {Path(emb_file).name} "
                                 f"(expected: {valid_embeddings[0].shape[1]}, got: {embeddings.shape[1]})")
                    skipped_files.append({
                        'file': emb_file,
                        'reason': 'dimension_mismatch',
                        'expected_dim': valid_embeddings[0].shape[1],
                        'actual_dim': embeddings.shape[1]
                    })
                    combined_summary['skipped_files'].append({
                        'embeddings': emb_file,
                        'metadata': meta_file,
                        'reason': 'dimension_mismatch',
                        'expected_dim': valid_embeddings[0].shape[1],
                        'actual_dim': embeddings.shape[1]
                    })
                    continue
                
                # File is valid - add to collection
                valid_embeddings.append(embeddings)
                all_metadata.extend(metadata)
                
                # Update combined summary
                combined_summary['source_files'].append({
                    'embeddings': emb_file,
                    'metadata': meta_file,
                    'summary': summary_file,
                    'num_embeddings': len(embeddings),
                    'num_chunks': len(metadata),
                    'dimensions': embeddings.shape[1]
                })
                
                combined_summary['total_embeddings'] += len(embeddings)
                combined_summary['total_chunks'] += len(metadata)
                
                # Collect unique documents and spaces
                for meta in metadata:
                    combined_summary['unique_documents'].add(meta.get('doc_id', 'unknown'))
                    if meta.get('space'):
                        combined_summary['unique_spaces'].add(meta['space'])
                
                logger.info(f"✅ Added {len(embeddings)} embeddings from {Path(emb_file).name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {Path(emb_file).name}: {e}")
                skipped_files.append({
                    'file': emb_file,
                    'reason': 'processing_error',
                    'error': str(e)
                })
                combined_summary['skipped_files'].append({
                    'embeddings': emb_file,
                    'metadata': meta_file,
                    'reason': 'processing_error',
                    'error': str(e)
                })
                continue
        
        # INTELLIGENT VALIDATION OF RESULTS
        if not valid_embeddings:
            raise ValueError(f"❌ No valid embedding files found. Processed {len(file_pairs)} files, "
                            f"skipped {len(skipped_files)} files. Check your embedding generation process.")
        
        # Combine all valid embeddings
        final_embeddings = np.vstack(valid_embeddings)
        
        # Convert sets to counts for JSON serialization
        combined_summary['unique_documents'] = len(combined_summary['unique_documents'])
        combined_summary['unique_spaces'] = len(combined_summary['unique_spaces'])
        combined_summary['files_processed'] = len(valid_embeddings)
        combined_summary['files_skipped'] = len(skipped_files)
        
        # INTELLIGENT SUMMARY REPORTING
        logger.info(f"🎯 Intelligent combination complete:")
        logger.info(f"   ✅ Valid files: {len(valid_embeddings)}")
        logger.info(f"   ⚠️  Skipped files: {len(skipped_files)}")
        logger.info(f"   📊 Total embeddings: {len(final_embeddings):,}")
        logger.info(f"   📄 Unique documents: {combined_summary['unique_documents']:,}")
        
        if skipped_files:
            logger.info("📋 Skipped files summary:")
            for skipped in skipped_files:
                logger.info(f"   • {Path(skipped['file']).name}: {skipped['reason']}")
        
        return final_embeddings, all_metadata, combined_summary
    
    def select_optimal_index_type(self, num_vectors: int, dimensions: int = None) -> str:
        """Intelligently select index type based on dataset characteristics"""
        
        # Memory estimation
        memory_gb = (num_vectors * (dimensions or 384) * 4) / (1024**3)
        
        logger.info(f"🧠 Intelligent index selection for {num_vectors:,} vectors:")
        logger.info(f"   📊 Estimated memory: {memory_gb:.2f} GB")
        
        if num_vectors < 1000:
            logger.info("   🎯 Selected: Flat index (small dataset, exact search)")
            return 'flat'
        elif num_vectors < 50000:
            logger.info("   🎯 Selected: HNSW index (medium dataset, fast approximate search)")
            return 'hnsw_small'
        elif num_vectors < 5000000:
            logger.info("   🎯 Selected: HNSW index (large dataset, optimized parameters)")
            return 'hnsw_large'
        else:
            logger.info("   🎯 Selected: IVF-PQ index (very large dataset, memory efficient)")
            return 'ivf_pq'
    
    def create_hnsw_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                         index_name: str = 'confluence_hnsw',
                         M: int = None, efConstruction: int = None) -> str:
        """Create optimized HNSW index with automatic parameter selection"""
        
        num_vectors = len(embeddings)
        
        # Auto-select parameters based on dataset size
        if M is None:
            if num_vectors < 10000:
                M = 16  # Smaller M for small datasets
            elif num_vectors < 100000:
                M = 32  # Standard M for medium datasets
            else:
                M = 64  # Larger M for large datasets
        
        if efConstruction is None:
            if num_vectors < 10000:
                efConstruction = 200
            elif num_vectors < 100000:
                efConstruction = 400
            else:
                efConstruction = 800  # Higher efConstruction for large datasets
        
        logger.info(f"Creating HNSW index with {num_vectors} vectors")
        logger.info(f"Optimized parameters: M={M}, efConstruction={efConstruction}")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(embeddings_normalized)
        
        dimension = embeddings.shape[1]
        
        # Create HNSW index
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = efConstruction
        
        # Set optimal number of threads
        num_threads = min(8, max(1, num_vectors // 10000))
        faiss.omp_set_num_threads(num_threads)
        logger.info(f"Using {num_threads} threads for index construction")
        
        # Add vectors in batches for memory efficiency
        batch_size = min(10000, max(1000, num_vectors // 10))
        logger.info(f"Adding vectors in batches of {batch_size}")
        
        start_time = time.time()
        for i in range(0, len(embeddings_normalized), batch_size):
            end_idx = min(i + batch_size, len(embeddings_normalized))
            batch = embeddings_normalized[i:end_idx]
            index.add(batch)
            
            if i % (batch_size * 10) == 0:  # Log every 10 batches
                elapsed = time.time() - start_time
                progress = (end_idx / len(embeddings_normalized)) * 100
                logger.info(f"Progress: {progress:.1f}% ({end_idx}/{len(embeddings_normalized)}) in {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Index construction completed in {total_time:.1f}s")
        
        return self._save_index(index, metadata, index_name, 'hnsw', {
            'M': M,
            'efConstruction': efConstruction,
            'num_threads': num_threads,
            'batch_size': batch_size,
            'construction_time_seconds': total_time
        })
    
    def create_flat_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                         index_name: str = 'confluence_flat') -> str:
        """Create flat index for exact search (recommended for small datasets only)"""
        
        num_vectors = len(embeddings)
        if num_vectors > 50000:
            logger.warning(f"Large dataset ({num_vectors} vectors) - HNSW index recommended for better performance")
        
        logger.info(f"Creating flat index with {num_vectors} vectors")
        
        # Normalize embeddings
        embeddings_normalized = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(embeddings_normalized)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Set parallel processing
        faiss.omp_set_num_threads(min(8, max(1, num_vectors // 5000)))
        
        start_time = time.time()
        index.add(embeddings_normalized)
        construction_time = time.time() - start_time
        
        logger.info(f"Flat index construction completed in {construction_time:.1f}s")
        
        return self._save_index(index, metadata, index_name, 'flat', {
            'construction_time_seconds': construction_time
        })
    
    def create_auto_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                         index_name: str = 'confluence_auto') -> str:
        """Automatically create the optimal index type based on dataset size"""
        
        num_vectors = len(embeddings)
        index_type = self.select_optimal_index_type(num_vectors, embeddings.shape[1])
        
        logger.info(f"Auto-selected index type: {index_type} for {num_vectors} vectors")
        
        if index_type == 'flat':
            return self.create_flat_index(embeddings, metadata, f"{index_name}_flat")
        else:
            # Use HNSW for both small and large datasets (as requested)
            return self.create_hnsw_index(embeddings, metadata, f"{index_name}_hnsw")
    
    def _save_index(self, index, metadata: List[Dict[str, Any]], 
                   index_name: str, index_type: str, parameters: Dict = None) -> str:
        """Save index, metadata, and comprehensive information"""
        
        # Create index-specific directory
        index_path = self.index_dir / index_name
        index_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_file = index_path / 'index.faiss'
        faiss.write_index(index, str(index_file))
        
        # Save metadata
        metadata_file = index_path / 'metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Create comprehensive index information
        index_info = {
            'index_name': index_name,
            'index_type': index_type,
            'dimension': index.d,
            'num_vectors': index.ntotal,
            'num_documents': len(set(m['doc_id'] for m in metadata)),
            'num_chunks': len(metadata),
            'creation_timestamp': datetime.now().isoformat(),
            'memory_usage_mb': self._estimate_memory_usage(index),
            'parameters': parameters or {},
            'files': {
                'index': str(index_file),
                'metadata': str(metadata_file)
            },
            'search_optimization': {
                'efSearch': 64 if index_type == 'hnsw' else None,
                'recommended_k': min(100, max(10, index.ntotal // 1000))
            },
            'statistics': {
                'unique_spaces': len(set(m.get('space', '') for m in metadata if m.get('space'))),
                'avg_chunk_length': np.mean([m.get('chunk_length', 0) for m in metadata]),
                'total_word_count': sum(m.get('word_count', 0) for m in metadata if 'word_count' in m)
            }
        }
        
        # Save index information
        info_file = index_path / 'index_info.json'
        with open(info_file, 'w') as f:
            json.dump(index_info, f, indent=2)
        
        logger.info(f"Index saved to: {index_path}")
        logger.info(f"Type: {index_type}, Vectors: {index_info['num_vectors']:,}, "
                   f"Documents: {index_info['num_documents']:,}, "
                   f"Memory: {index_info['memory_usage_mb']:.1f} MB")
        
        return str(index_path)
    
    def _estimate_memory_usage(self, index) -> float:
        """Estimate memory usage of index in MB"""
        try:
            if hasattr(index, 'ntotal') and hasattr(index, 'd'):
                if isinstance(index, faiss.IndexFlatIP):
                    return index.ntotal * index.d * 4 / (1024 * 1024)  # 4 bytes per float
                elif isinstance(index, faiss.IndexHNSWFlat):
                    # HNSW has additional graph structure overhead
                    return index.ntotal * index.d * 4 * 1.8 / (1024 * 1024)
                else:
                    return index.ntotal * index.d * 4 / (1024 * 1024)
        except:
            return 0.0
    
    def load_index(self, index_path: str) -> bool:
        """Load an existing index for testing or analysis"""
        index_dir = Path(index_path)
        
        if not index_dir.exists():
            logger.error(f"Index directory not found: {index_path}")
            return False
        
        index_file = index_dir / 'index.faiss'
        metadata_file = index_dir / 'metadata.pkl'
        info_file = index_dir / 'index_info.json'
        
        if not all(f.exists() for f in [index_file, metadata_file, info_file]):
            logger.error(f"Missing required files in {index_path}")
            return False
        
        try:
            # Load index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load info
            with open(info_file, 'r') as f:
                self.index_info = json.load(f)
            
            # Optimize for search
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = self.index_info.get('search_optimization', {}).get('efSearch', 64)
            
            logger.info(f"Loaded {self.index_info['index_type']} index: "
                       f"{self.index_info['num_vectors']:,} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def list_available_indexes(self) -> List[Dict[str, Any]]:
        """List all available indexes"""
        indexes = []
        
        for index_dir in self.index_dir.iterdir():
            if index_dir.is_dir():
                info_file = index_dir / 'index_info.json'
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        info['path'] = str(index_dir)
                        indexes.append(info)
                    except Exception as e:
                        logger.warning(f"Could not read index info from {index_dir}: {e}")
        
        # Sort by creation time (newest first)
        indexes.sort(key=lambda x: x.get('creation_timestamp', ''), reverse=True)
        return indexes

def main():
    parser = argparse.ArgumentParser(description='Enhanced Confluence FAISS Indexer')
    
    # Input options
    parser.add_argument('--embeddings-dir', default=str(EMBEDDINGS_DIR),
                       help='Directory containing embedding files')
    parser.add_argument('--embeddings-file',
                       help='Specific embeddings .npy file')
    parser.add_argument('--metadata-file',
                       help='Specific metadata .pkl file')
    parser.add_argument('--summary-file',
                       help='Optional summary .json file')
    
    # Index options
    parser.add_argument('--index-dir', default=str(INDEXES_DIR),
                       help='Directory to save indexes')
    parser.add_argument('--index-name', default='confluence',
                       help='Base name for the index')
    parser.add_argument('--index-type', 
                       choices=['auto', 'hnsw', 'flat'],
                       default='auto',
                       help='Type of index to create (auto recommended)')
    
    # HNSW parameters
    parser.add_argument('--hnsw-M', type=int,
                       help='HNSW M parameter (auto-selected if not specified)')
    parser.add_argument('--hnsw-ef-construction', type=int,
                       help='HNSW efConstruction parameter (auto-selected if not specified)')
    
    # Processing options
    parser.add_argument('--combine-files', action='store_true',
                       help='Combine multiple embedding files into single index')
    parser.add_argument('--force-hnsw', action='store_true',
                       help='Force HNSW index even for small datasets')
    
    # Utility options
    parser.add_argument('--list-embeddings', action='store_true',
                       help='List available embedding files and exit')
    parser.add_argument('--list-indexes', action='store_true',
                       help='List available indexes and exit')
    parser.add_argument('--test-index',
                       help='Test load an existing index')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize indexer
    indexer = EnhancedConfluenceIndexer(args.index_dir)
    
    try:
        # List embeddings if requested
        if args.list_embeddings:
            try:
                file_pairs = indexer.discover_embedding_files(args.embeddings_dir)
                print(f"\n📁 Found {len(file_pairs)} embedding file pairs:")
                for i, (emb_file, meta_file, summary_file) in enumerate(file_pairs, 1):
                    print(f"  {i:2d}: {Path(emb_file).name}")
                    print(f"      Metadata: {Path(meta_file).name}")
                    if Path(summary_file).exists():
                        print(f"      Summary: {Path(summary_file).name}")
                return 0
            except Exception as e:
                print(f"❌ Error listing embeddings: {e}")
                return 1
        
        # List indexes if requested
        if args.list_indexes:
            indexes = indexer.list_available_indexes()
            if indexes:
                print(f"\n📚 Available indexes:")
                for idx in indexes:
                    print(f"  • {idx['index_name']} ({idx['index_type']}) - "
                          f"{idx['num_vectors']:,} vectors, "
                          f"{idx.get('memory_usage_mb', 0):.1f} MB")
                    if 'creation_timestamp' in idx:
                        print(f"    Created: {idx['creation_timestamp']}")
            else:
                print("❌ No indexes found")
            return 0
        
        # Test index loading if requested
        if args.test_index:
            success = indexer.load_index(args.test_index)
            if success:
                print(f"✅ Successfully loaded index: {args.test_index}")
                print(f"📊 Index info: {indexer.index_info}")
            else:
                print(f"❌ Failed to load index: {args.test_index}")
            return 0 if success else 1
        
        # INTELLIGENT FILE PROCESSING
        if args.embeddings_file and args.metadata_file:
            # Process specific files with validation
            logger.info("🎯 Processing specific files with intelligent validation")
            embeddings, metadata, summary = indexer.load_embedding_data(
                args.embeddings_file, args.metadata_file, args.summary_file
            )
            
            # Validate loaded data
            if len(embeddings) == 0:
                logger.error("❌ Specified embeddings file is empty")
                return 1
                
        else:
            # INTELLIGENT AUTO-DISCOVERY AND COMBINATION
            logger.info("🔍 Intelligent auto-discovery mode")
            file_pairs = indexer.discover_embedding_files(args.embeddings_dir)
            
            if len(file_pairs) == 1:
                logger.info("📄 Single valid file pair found - processing directly")
                embeddings, metadata, summary = indexer.load_embedding_data(*file_pairs[0])
            else:
                logger.info(f"📚 Multiple valid files found ({len(file_pairs)}) - intelligent combination")
                embeddings, metadata, summary = indexer.combine_multiple_embeddings(file_pairs)
        
        # Create index
        logger.info(f"Creating {args.index_type} index for {len(embeddings):,} vectors")
        
        if args.index_type == 'auto':
            if args.force_hnsw:
                index_path = indexer.create_hnsw_index(
                    embeddings, metadata, f"{args.index_name}_hnsw",
                    args.hnsw_M, args.hnsw_ef_construction
                )
            else:
                index_path = indexer.create_auto_index(embeddings, metadata, args.index_name)
        elif args.index_type == 'hnsw':
            index_path = indexer.create_hnsw_index(
                embeddings, metadata, f"{args.index_name}_hnsw",
                args.hnsw_M, args.hnsw_ef_construction
            )
        elif args.index_type == 'flat':
            index_path = indexer.create_flat_index(embeddings, metadata, f"{args.index_name}_flat")
        
        # Success summary
        print(f"\n✅ Intelligent indexing complete!")
        print(f"🎯 Strategy: Automatically handled empty files and optimized parameters")
        print(f"📊 Dataset: {len(embeddings):,} vectors from {len(set(m['doc_id'] for m in metadata)):,} documents")
        print(f"📁 Index saved to: {index_path}")
        print(f"🔍 Ready for search operations")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Intelligent indexing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
