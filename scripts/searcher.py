#!/usr/bin/env python3

"""
Enhanced Confluence Searcher with Mistral 7B Integration for macOS
Supports batched data processing and scalable FAISS indexes
Updated for intelligent indexing system
"""

import argparse
import json
import logging
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Project structure paths - Updated to match new structure
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEXES_DIR = DATA_DIR / "indexes"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = EMBEDDINGS_DIR / "cache"
CHECKPOINT_DIR = EMBEDDINGS_DIR / "checkpoints"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with comprehensive metadata"""
    id: str
    title: str
    content: str
    snippet: str
    score: Optional[float]
    match_type: str
    space: str = ""
    creator: str = ""
    last_modified: str = ""
    attachments: List[str] = None
    explanation: str = ""
    chunk_id: int = 0
    total_chunks: int = 1
    word_count: int = 0
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []

class EnhancedConfluenceSearcher:
    """Advanced Confluence searcher with Mistral integration and intelligent index support"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 score_threshold: float = 0.3,
                 enable_mistral: bool = True,
                 mistral_script_path: str = None):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.enable_mistral = enable_mistral
        self.mistral_script_path = mistral_script_path or str(SCRIPTS_DIR / "mistral_service.py")
        
        # Core components
        self.model = None
        self.index = None
        self.metadata = None
        self.documents = None
        self.index_info = None
        
        # Performance tracking
        self.search_count = 0
        self.mistral_calls = 0
        self.cache_hits = 0
        
        # Document lookup optimization
        self.doc_lookup = {}
        self.title_lookup = {}
        
        # Ensure directories exist
        for directory in [LOGS_DIR, EMBEDDINGS_DIR, INDEXES_DIR, PARSED_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
        return True
    
    def auto_discover_indexes(self) -> List[Dict[str, Any]]:
        """Auto-discover available indexes using the intelligent indexer structure"""
        indexes = []
        
        for index_dir in INDEXES_DIR.iterdir():
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
    
    def load_latest_index(self) -> bool:
        """Automatically load the most recent index"""
        indexes = self.auto_discover_indexes()
        if not indexes:
            logger.error("No indexes found in the indexes directory")
            return False
        
        latest_index = indexes[0]
        logger.info(f"Auto-loading latest index: {latest_index['index_name']} "
                   f"({latest_index['index_type']}, {latest_index['num_vectors']} vectors)")
        
        return self.load_index(latest_index['path'])
    
    def load_index(self, index_path: str) -> bool:
        """Load FAISS index and metadata from intelligent indexer output"""
        index_dir = Path(index_path)
        
        if not index_dir.exists():
            logger.error(f"Index directory not found: {index_path}")
            return False
        
        index_file = index_dir / 'index.faiss'
        metadata_file = index_dir / 'metadata.pkl'
        info_file = index_dir / 'index_info.json'
        
        if not all(f.exists() for f in [index_file, metadata_file, info_file]):
            logger.error(f"Missing index files in {index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load index info
            with open(info_file, 'r') as f:
                self.index_info = json.load(f)
            
            # Optimize index for search
            self._optimize_index_for_search()
            
            logger.info(f"Loaded {self.index_info['index_type']} index: "
                       f"{self.index_info['num_vectors']:,} vectors, "
                       f"{self.index_info['num_documents']:,} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _optimize_index_for_search(self):
        """Optimize index parameters for search performance"""
        if hasattr(self.index, 'nprobe'):
            # For IVF indexes
            nlist = getattr(self.index, 'nlist', 100)
            self.index.nprobe = min(max(10, nlist // 10), 100)
            logger.info(f"Set nprobe to {self.index.nprobe}")
        
        if hasattr(self.index, 'hnsw'):
            # For HNSW indexes - use optimized efSearch from index info
            ef_search = self.index_info.get('search_optimization', {}).get('efSearch', 64)
            self.index.hnsw.efSearch = ef_search
            logger.info(f"Set efSearch to {ef_search}")
    
    def auto_discover_documents(self) -> bool:
        """Auto-discover and load documents from parsed directory"""
        # Try to load batch files first
        batch_files = sorted(PARSED_DIR.glob('confluence_batch_*.json'))
        batch_files = [f for f in batch_files if 'manifest' not in f.name.lower()]
        
        if batch_files:
            logger.info(f"Found {len(batch_files)} batch files, loading...")
            return self.load_batch_documents(str(PARSED_DIR))
        
        # Try to load single JSON files
        json_files = list(PARSED_DIR.glob('*.json'))
        json_files = [f for f in json_files if 'manifest' not in f.name.lower()]
        
        if json_files:
            logger.info(f"Found {len(json_files)} JSON files, loading first one...")
            return self.load_documents(str(json_files[0]))
        
        logger.error("No document files found in parsed directory")
        return False
    
    def load_documents(self, documents_path: str) -> bool:
        """Load original documents for full content access"""
        try:
            documents_file = Path(documents_path)
            
            # Handle different document formats
            if documents_file.suffix == '.json':
                with open(documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            elif documents_file.suffix == '.pkl':
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                # Try to find JSON files in the directory
                if documents_file.is_dir():
                    json_files = list(documents_file.glob('*.json'))
                    json_files = [f for f in json_files if 'manifest' not in f.name.lower()]
                    if json_files:
                        # Load the first JSON file found
                        with open(json_files[0], 'r', encoding='utf-8') as f:
                            self.documents = json.load(f)
                    else:
                        logger.error(f"No JSON files found in {documents_path}")
                        return False
                else:
                    logger.error(f"Unsupported document format: {documents_file.suffix}")
                    return False
            
            # Create lookup dictionaries for fast access
            self.doc_lookup = {doc['id']: doc for doc in self.documents}
            self.title_lookup = {doc['title'].lower(): doc for doc in self.documents}
            
            logger.info(f"Loaded {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return False
    
    def load_batch_documents(self, batch_dir: str, pattern: str = "confluence_batch_*.json") -> bool:
        """Load documents from multiple batch files with intelligent filtering"""
        batch_path = Path(batch_dir)
        all_files = sorted(batch_path.glob(pattern))
        
        # INTELLIGENT FILE FILTERING
        batch_files = []
        for file_path in all_files:
            filename = file_path.name.lower()
            # Skip manifest files and other non-document files
            if any(skip_word in filename for skip_word in ['manifest', 'summary', 'metadata']):
                logger.info(f"Skipping non-document file: {file_path.name}")
                continue
            batch_files.append(file_path)
        
        if not batch_files:
            logger.error(f"No valid batch files found matching {pattern} in {batch_dir}")
            return False
        
        logger.info(f"Processing {len(batch_files)} batch files")
        
        all_documents = []
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different data structures
                if isinstance(data, list):
                    # Direct list of documents
                    batch_docs = data
                elif isinstance(data, dict) and 'results' in data:
                    # Wrapped in results object
                    batch_docs = data['results']
                elif isinstance(data, dict) and 'documents' in data:
                    # Wrapped in documents object
                    batch_docs = data['documents']
                else:
                    logger.warning(f"Skipping {batch_file}: unrecognized structure")
                    continue
                
                # Validate and filter documents
                valid_docs = []
                for doc in batch_docs:
                    if isinstance(doc, dict) and 'id' in doc and 'title' in doc:
                        valid_docs.append(doc)
                    else:
                        logger.debug(f"Skipping invalid document structure in {batch_file}")
                
                all_documents.extend(valid_docs)
                logger.info(f"Loaded {len(valid_docs)} valid documents from {batch_file}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {batch_file}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to load batch file {batch_file}: {e}")
                continue
        
        if not all_documents:
            logger.error("No valid documents found in any batch files")
            return False
        
        self.documents = all_documents
        
        # Create lookup dictionaries with error handling
        try:
            self.doc_lookup = {doc['id']: doc for doc in self.documents if 'id' in doc}
            self.title_lookup = {doc['title'].lower(): doc for doc in self.documents if 'title' in doc}
        except Exception as e:
            logger.error(f"Failed to create document lookups: {e}")
            return False
        
        logger.info(f"Successfully loaded {len(self.documents)} total documents from {len(batch_files)} batch files")
        return True
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform semantic search using FAISS index"""
        if not self.model or not self.index or not self.metadata:
            logger.error("Model, index, or metadata not loaded")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding.astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                if score < self.score_threshold:
                    continue
                
                meta = self.metadata[idx]
                doc_id = meta['doc_id']
                
                # Get full document for context
                full_doc = self.doc_lookup.get(doc_id, {})
                
                # Create snippet from chunk
                snippet = self._create_snippet(meta.get('chunk_text', ''), query)
                
                result = SearchResult(
                    id=doc_id,
                    title=meta.get('title', 'Untitled'),
                    content=full_doc.get('content', ''),
                    snippet=snippet,
                    score=float(score),
                    match_type='semantic',
                    space=meta.get('space', ''),
                    creator=meta.get('creator', ''),
                    last_modified=meta.get('lastModificationDate', ''),
                    attachments=meta.get('attachments', []),
                    explanation=f"Semantic similarity: {score:.3f}",
                    chunk_id=meta.get('chunk_id', 0),
                    total_chunks=meta.get('total_chunks', 1),
                    word_count=meta.get('word_count', 0)
                )
                
                results.append(result)
            
            self.search_count += 1
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def boolean_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform boolean/keyword search"""
        if not self.documents:
            logger.error("Documents not loaded")
            return []
        
        # Parse boolean query
        terms = self._parse_boolean_query(query)
        results = []
        
        for doc in self.documents:
            score = self._calculate_boolean_score(doc, terms)
            if score > 0:
                snippet = self._create_snippet(doc.get('content', ''), query)
                
                result = SearchResult(
                    id=doc['id'],
                    title=doc.get('title', 'Untitled'),
                    content=doc.get('content', ''),
                    snippet=snippet,
                    score=score,
                    match_type='boolean',
                    space=doc.get('space', ''),
                    creator=doc.get('creator', ''),
                    last_modified=doc.get('lastModificationDate', ''),
                    attachments=doc.get('attachments', []),
                    explanation=f"Boolean match score: {score}",
                    word_count=doc.get('word_count', 0)
                )
                
                results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def smart_search(self, query: str, top_k: int = 10,
                    filters: Dict[str, str] = None) -> List[SearchResult]:
        """Intelligent search combining semantic and boolean approaches"""
        if self._is_boolean_query(query):
            results = self.boolean_search(query, top_k)
        else:
            results = self.semantic_search(query, top_k)
        
        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)
        
        return results
    
    def _is_boolean_query(self, query: str) -> bool:
        """Detect if query contains boolean operators"""
        boolean_operators = ['AND', 'OR', 'NOT', '+', '-', '"']
        return any(op in query.upper() for op in boolean_operators)
    
    def _parse_boolean_query(self, query: str) -> Dict[str, List[str]]:
        """Parse boolean query into terms"""
        # Simplified boolean parsing
        terms = {
            'must_have': [],
            'should_have': [],
            'must_not_have': [],
            'exact_phrases': []
        }
        
        # Extract exact phrases
        phrases = re.findall(r'"([^"]*)"', query)
        terms['exact_phrases'] = phrases
        
        # Remove phrases from query
        query_no_phrases = re.sub(r'"[^"]*"', '', query)
        
        # Split into words and process
        words = query_no_phrases.split()
        for word in words:
            if word.startswith('+'):
                terms['must_have'].append(word[1:].lower())
            elif word.startswith('-'):
                terms['must_not_have'].append(word[1:].lower())
            elif word.upper() not in ['AND', 'OR', 'NOT']:
                terms['should_have'].append(word.lower())
        
        return terms
    
    def _calculate_boolean_score(self, doc: Dict[str, Any], terms: Dict[str, List[str]]) -> float:
        """Calculate boolean match score"""
        content = (doc.get('title', '') + ' ' + doc.get('content', '')).lower()
        score = 0.0
        
        # Must have terms
        for term in terms['must_have']:
            if term in content:
                score += 2.0
            else:
                return 0.0  # Must have all required terms
        
        # Should have terms
        for term in terms['should_have']:
            if term in content:
                score += 1.0
        
        # Must not have terms
        for term in terms['must_not_have']:
            if term in content:
                return 0.0  # Exclude if contains forbidden terms
        
        # Exact phrases
        for phrase in terms['exact_phrases']:
            if phrase.lower() in content:
                score += 3.0
            else:
                return 0.0  # Must have all exact phrases
        
        return score
    
    def _create_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Create a snippet highlighting query terms"""
        if not text:
            return ""
        
        # Find the best position to start the snippet
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        best_pos = 0
        max_matches = 0
        
        # Look for position with most query term matches
        for i in range(0, len(text) - max_length, 50):
            snippet_text = text_lower[i:i + max_length]
            matches = sum(1 for term in query_terms if term in snippet_text)
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        # Extract snippet
        snippet = text[best_pos:best_pos + max_length]
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + max_length < len(text):
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, str]) -> List[SearchResult]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            include = True
            
            if 'space' in filters:
                if filters['space'].lower() not in result.space.lower():
                    include = False
            
            if 'creator' in filters:
                if filters['creator'].lower() not in result.creator.lower():
                    include = False
            
            if 'date' in filters:
                # Simple date filtering (could be enhanced)
                if filters['date'] not in result.last_modified:
                    include = False
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    # Mistral Integration Methods
    def call_mistral(self, prompt: str, context: str = None) -> str:
        """Call Mistral service with prompt and context"""
        if not self.enable_mistral:
            return "❌ Mistral integration not enabled"
        
        print(f"DEBUG: Calling Mistral with prompt='{prompt}', context='{context}'")
        
        try:
            cmd = [
                'python3', self.mistral_script_path,
                '--prompt', prompt
            ]
            
            if context:
                cmd.extend(['--context', context])
            
            print(f"DEBUG: Command = {cmd}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            print(f"DEBUG: Return code = {result.returncode}")
            print(f"DEBUG: Stdout = '{result.stdout}'")
            print(f"DEBUG: Stderr = '{result.stderr}'")
            
            if result.returncode == 0:
                self.mistral_calls += 1
                return result.stdout.strip()
            else:
                logger.error(f"Mistral call failed: {result.stderr}")
                return f"❌ Mistral error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ Mistral call timed out"
        except Exception as e:
            logger.error(f"Failed to call Mistral: {e}")
            return f"❌ Failed to call Mistral: {e}"
    
    def summarize_page(self, page_id: str = None, title: str = None) -> str:
        """Summarize a specific page using Mistral"""
        doc = None
        if page_id:
            doc = self.doc_lookup.get(page_id)
        elif title:
            doc = self.title_lookup.get(title.lower())
        
        if not doc:
            return f"❌ Page not found: {page_id or title}"
        
        content = doc.get('content', '')
        if not content:
            return f"❌ No content found for page: {doc.get('title', 'Unknown')}"
        
        # Limit content length for Mistral
        if len(content) > 1500:
            content = content[:1500] + "..."
        
        prompt = f"Summarize this Confluence page titled '{doc.get('title', 'Unknown')}':"
        return self.call_mistral(prompt, content)
    
    def improve_section(self, page_id: str = None, title: str = None,
                       section_keyword: str = None) -> str:
        """Improve a section of a page using Mistral"""
        doc = None
        if page_id:
            doc = self.doc_lookup.get(page_id)
        elif title:
            doc = self.title_lookup.get(title.lower())
        
        if not doc:
            return f"❌ Page not found: {page_id or title}"
        
        content = doc.get('content', '')
        if not content:
            return f"❌ No content found for page: {doc.get('title', 'Unknown')}"
        
        # Extract section if keyword provided
        if section_keyword:
            section_content = self._extract_section(content, section_keyword)
            if section_content:
                content = section_content
        
        # Limit content length
        if len(content) > 1200:
            content = content[:1200] + "..."
        
        section_part = f" (section: {section_keyword})" if section_keyword else ""
        prompt = f"Improve this content from '{doc.get('title', 'Unknown')}'{section_part}:"
        
        return self.call_mistral(prompt, content)
    
    def answer_question(self, question: str, max_context_docs: int = 3) -> str:
        """Answer a question using relevant documents as context"""
        # Search for relevant documents
        search_results = self.smart_search(question, top_k=max_context_docs)
        
        if not search_results:
            return f"❌ No relevant documents found for: {question}"
        
        # Combine content from top results as context
        context_parts = []
        for result in search_results:
            context_parts.append(f"From '{result.title}':\n{result.snippet}")
        
        context = "\n\n".join(context_parts)
        
        # Limit context length
        if len(context) > 1200:
            context = context[:1200] + "..."
        
        prompt = f"Answer this question based on the Confluence documentation: {question}"
        return self.call_mistral(prompt, context)
    
    def _extract_section(self, content: str, keyword: str) -> str:
        """Extract section containing keyword"""
        lines = content.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            if keyword.lower() in line.lower():
                in_section = True
                section_lines.append(line)
            elif in_section:
                if line.strip() and not line.startswith(' '):
                    # Potential new section
                    if len(section_lines) > 5:  # Got enough content
                        break
                section_lines.append(line)
        
        return '\n'.join(section_lines) if section_lines else ""
    
    # Standalone text processing methods
    def summarize_text(self, text: str) -> str:
        """Summarize arbitrary text"""
        if len(text) > 1200:
            text = text[:1200] + "..."
        
        prompt = "Summarize this text:"
        print(f"DEBUG: Searcher prompt = '{prompt}, text: {text}'")
        return self.call_mistral(prompt, text)
    
    def rewrite_text(self, text: str, instruction: str = None) -> str:
        """Rewrite text with optional instruction"""
        if len(text) > 1200:
            text = text[:1200] + "..."
        
        base_prompt = "Rewrite this text for clarity:"
        if instruction:
            base_prompt = f"Rewrite this text with instruction '{instruction}':"
        
        return self.call_mistral(base_prompt, text)
    
    def improve_text(self, text: str) -> str:
        """Improve text structure and clarity"""
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        prompt = "Improve this text:"
        return self.call_mistral(prompt, text)
    
    def answer_with_context(self, question: str, context: str) -> str:
        """Answer question with provided context"""
        if len(context) > 1000:
            context = context[:1000] + "..."
        
        prompt = f"Answer this question: {question}"
        return self.call_mistral(prompt, context)
    
    def process_custom_prompt(self, prompt: str, context: str = None) -> str:
        """Process custom prompt with optional context"""
        return self.call_mistral(prompt, context)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search system statistics"""
        stats = {
            'total_searches': self.search_count,
            'mistral_calls': self.mistral_calls,
            'cache_hits': self.cache_hits,
            'documents_loaded': len(self.documents) if self.documents else 0,
            'index_type': self.index_info.get('index_type', 'unknown') if self.index_info else 'not_loaded',
            'model_name': self.model_name,
            'mistral_enabled': self.enable_mistral
        }
        
        if self.index_info:
            stats.update({
                'index_vectors': self.index_info.get('num_vectors', 0),
                'index_documents': self.index_info.get('num_documents', 0),
                'index_creation_time': self.index_info.get('creation_timestamp', 'unknown')
            })
        
        return stats

def get_user_input(prompt_text: str) -> str:
    """Get user input with proper handling"""
    try:
        return input(f"\n{prompt_text}: ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

def enhanced_interactive_search(searcher: EnhancedConfluenceSearcher):
    """Enhanced interactive search interface"""
    print("🔍 Enhanced Confluence Search with Mistral 7B Integration")
    print(f"📚 Loaded: {len(searcher.documents)} documents")
    print(f"🤖 Mistral: {'Enabled' if searcher.enable_mistral else 'Disabled'}")
    
    if searcher.index_info:
        print(f"📊 Index: {searcher.index_info['index_type']} with {searcher.index_info['num_vectors']:,} vectors")
    
    if searcher.enable_mistral:
        print("\n🎯 Mistral Commands:")
        print("  • summarize <page_id> or summarize: <page_id> or summarize title:<title>")
        print("  • improve <page_id> [section:<keyword>]")
        print("  • ask <question>")
        print("  • summarize: <text> - summarize arbitrary text")
        print("  • rewrite: <text> - rewrite text")
        print("  • improve: <text> - improve text")
        print("  • answer: <question> --context <context>")
        print("  • prompt: <custom_prompt>")
    
    print("\n🔧 Search Features:")
    print("  • Natural language queries")
    print("  • Boolean search (AND, OR, NOT, +, -, \"exact phrases\")")
    print("  • Filters: space:name, creator:name, date:YYYY")
    print("  • 'stats' - show system statistics")
    print("  • 'indexes' - list available indexes")
    print("  • 'quit' - exit")
    
    while True:
        try:
            query = get_user_input("\n🔍 Search query (or command)")
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'stats':
                stats = searcher.get_search_stats()
                print("\n📊 Search System Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if query.lower() == 'indexes':
                indexes = searcher.auto_discover_indexes()
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
                continue
            
            if not query:
                continue
            
            # Handle Mistral commands if enabled
            if searcher.enable_mistral:
                # ENHANCED SUMMARIZE COMMAND HANDLING - Accepts both formats
                if query.startswith('summarize'):
                    # Strip optional colon and get the parameter
                    if query.startswith('summarize:'):
                        param = query[10:].strip()
                    elif query.startswith('summarize '):
                        param = query[10:].strip()
                    else:
                        param = ""
                    
                    if param:
                        # Handle quoted text (strip quotes if present)
                        if (param.startswith('"') and param.endswith('"')) or (param.startswith("'") and param.endswith("'")):
                            # It's quoted text - treat as arbitrary text to summarize
                            text_to_summarize = param[1:-1]  # Remove quotes
                            result = searcher.summarize_text(text_to_summarize)
                            print(f"\n{result}")
                        elif param.isdigit():
                            # It's a document ID
                            doc = searcher.doc_lookup.get(param)
                            if doc:
                                content = doc.get('content', '')
                                title = doc.get('title', 'Unknown Document')
                                if content:
                                    # Create a formatted prompt for better summarization
                                    formatted_content = f"Title: {title}\n\nContent: {content}"
                                    result = searcher.call_mistral(
                                        "Summarize this Confluence document in 2-3 sentences, focusing on key points:",
                                        formatted_content
                                    )
                                    print(f"\n📄 **Summary of '{title}'**\n\n{result}")
                                else:
                                    print(f"\n❌ No content found for document ID: {param}")
                            else:
                                print(f"\n❌ Document not found with ID: {param}")
                        elif param.startswith('title:'):
                            # It's a title lookup
                            title = param[6:].strip()
                            result = searcher.summarize_page(title=title)
                            print(f"\n{result}")
                        else:
                            # It's unquoted arbitrary text to summarize
                            result = searcher.summarize_text(param)
                            print(f"\n{result}")
                    else:
                        print("\n❌ Please provide a document ID, title, or text to summarize")
                    continue
                
                elif query.startswith('improve ') and not query.startswith('improve:'):
                    param = query[8:].strip()
                    if ' section:' in param:
                        page_part, section = param.split(' section:', 1)
                        if page_part.startswith('title:'):
                            title = page_part[6:].strip()
                            result = searcher.improve_section(title=title, section_keyword=section.strip())
                        else:
                            result = searcher.improve_section(page_id=page_part.strip(), section_keyword=section.strip())
                    else:
                        if param.startswith('title:'):
                            title = param[6:].strip()
                            result = searcher.improve_section(title=title)
                        else:
                            result = searcher.improve_section(page_id=param)
                    print(f"\n{result}")
                    continue
                
                elif query.startswith('ask '):
                    question = query[4:].strip()
                    result = searcher.answer_question(question)
                    print(f"\n{result}")
                    continue
                
                # Standalone text operations
                elif query.startswith('rewrite:'):
                    text = query[8:].strip()
                    instruction = get_user_input("Enter rewrite instruction (optional)")
                    result = searcher.rewrite_text(text, instruction if instruction else None)
                    print(f"\n{result}")
                    continue
                
                elif query.startswith('improve:'):
                    text = query[8:].strip()
                    result = searcher.improve_text(text)
                    print(f"\n{result}")
                    continue
                
                elif query.startswith('answer:'):
                    parts = query[7:].strip().split(' --context ')
                    if len(parts) == 2:
                        question, context = parts
                        result = searcher.answer_with_context(question.strip(), context.strip())
                    else:
                        question = parts[0]
                        context = get_user_input("Enter context")
                        result = searcher.answer_with_context(question.strip(), context)
                    print(f"\n{result}")
                    continue
                
                elif query.startswith('prompt:'):
                    custom_prompt = query[7:].strip()
                    context = get_user_input("Enter context (optional)")
                    result = searcher.process_custom_prompt(custom_prompt, context if context else None)
                    print(f"\n{result}")
                    continue
            
            # Regular search
            filters = {}
            for f in ['space:', 'creator:', 'date:']:
                if f in query.lower():
                    m = re.search(fr"{f}([^\s]+)", query, re.IGNORECASE)
                    if m:
                        filters[f[:-1]] = m.group(1)
            
            if searcher._is_boolean_query(query):
                print("🔧 Detected: Boolean query")
            else:
                print("💬 Detected: Natural language query")
            
            print("🔎 Searching...")
            results = searcher.smart_search(query, top_k=5, filters=filters)
            
            if not results:
                print("❌ No results found.")
                print("💡 Try different keywords, boolean operators, or Mistral commands")
                continue
            
            print(f"\n✅ Found {len(results)} result(s):\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.title}")
                print(f"   ID: {result.id}")
                print(f"   Match Type: {result.match_type.title()}")
                if result.score is not None and result.match_type != "boolean":
                    print(f"   Score: {result.score:.3f}")
                if result.space:
                    print(f"   Space: {result.space}")
                if result.creator:
                    print(f"   Creator: {result.creator}")
                if result.last_modified:
                    print(f"   Last Modified: {result.last_modified}")
                print(f"   → {result.snippet}")
                if result.attachments:
                    print(f"   📎 Attachments: {', '.join(result.attachments)}")
                print(f"   Explanation: {result.explanation}")
                print()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Search error: {e}")
            print("❌ Search error occurred. Please try again.")
    
    print("👋 Goodbye!")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Confluence Search with Mistral 7B Integration for macOS')
    
    parser.add_argument('--index-path', help='Path to FAISS index directory (auto-discovers if not provided)')
    parser.add_argument('--documents', help='Path to original documents JSON file (auto-discovers if not provided)')
    parser.add_argument('--batch-dir', help='Directory containing batch JSON files')
    parser.add_argument('--batch-pattern', default='confluence_batch_*.json', help='Pattern for batch files')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model name')
    parser.add_argument('--score-threshold', type=float, default=0.3, help='Minimum similarity score threshold')
    parser.add_argument('--disable-mistral', action='store_true', help='Disable Mistral integration')
    parser.add_argument('--mistral-script', help='Path to Mistral service script')
    
    # Search options
    parser.add_argument('--query', help='Direct search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--search-type', choices=['semantic', 'boolean', 'smart'], default='smart',
                       help='Type of search to perform')
    
    # Filter options
    parser.add_argument('--space', help='Filter by space name')
    parser.add_argument('--creator', help='Filter by creator name')
    parser.add_argument('--date', help='Filter by date (YYYY format)')
    
    # Mistral operations
    parser.add_argument('--summarize-page', help='Summarize specific page by ID')
    parser.add_argument('--summarize-title', help='Summarize specific page by title')
    parser.add_argument('--improve-page', help='Improve specific page by ID')
    parser.add_argument('--improve-title', help='Improve specific page by title')
    parser.add_argument('--section-keyword', help='Focus on section containing keyword')
    parser.add_argument('--ask-question', help='Ask a question using document context')
    
    # Standalone text operations
    parser.add_argument('--summarize-text', help='Summarize provided text')
    parser.add_argument('--rewrite-text', help='Rewrite provided text')
    parser.add_argument('--improve-text', help='Improve provided text')
    parser.add_argument('--answer-question', help='Answer question with provided context')
    parser.add_argument('--context', help='Context for question answering or custom prompts')
    parser.add_argument('--custom-prompt', help='Custom prompt to process')
    
    # Utility options
    parser.add_argument('--list-indexes', action='store_true', help='List available indexes')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    for directory in [LOGS_DIR, EMBEDDINGS_DIR, INDEXES_DIR, PARSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize searcher
    searcher = EnhancedConfluenceSearcher(
        model_name=args.model,
        score_threshold=args.score_threshold,
        enable_mistral=not args.disable_mistral,
        mistral_script_path=args.mistral_script
    )
    
    # Load model
    if not searcher.load_model():
        print("❌ Failed to load sentence transformer model")
        return 1
    
    # Handle list indexes command
    if args.list_indexes:
        indexes = searcher.auto_discover_indexes()
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
    
    # Load index
    if args.index_path:
        if not searcher.load_index(args.index_path):
            print("❌ Failed to load specified index")
            return 1
    else:
        if not searcher.load_latest_index():
            print("❌ Failed to auto-discover and load index")
            return 1
    
    # Load documents
    if args.documents:
        if not searcher.load_documents(args.documents):
            print("❌ Failed to load specified documents")
            return 1
    elif args.batch_dir:
        if not searcher.load_batch_documents(args.batch_dir, args.batch_pattern):
            print("❌ Failed to load batch documents")
            return 1
    else:
        if not searcher.auto_discover_documents():
            print("❌ Failed to auto-discover and load documents")
            return 1
    
    # Handle stats command
    if args.stats:
        stats = searcher.get_search_stats()
        print("\n📊 Search System Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0
    
    # Handle Mistral operations
    if searcher.enable_mistral:
        if args.summarize_page:
            result = searcher.summarize_page(page_id=args.summarize_page)
            print(f"\n{result}")
            return 0
        
        if args.summarize_title:
            result = searcher.summarize_page(title=args.summarize_title)
            print(f"\n{result}")
            return 0
        
        if args.improve_page:
            result = searcher.improve_section(page_id=args.improve_page, section_keyword=args.section_keyword)
            print(f"\n{result}")
            return 0
        
        if args.improve_title:
            result = searcher.improve_section(title=args.improve_title, section_keyword=args.section_keyword)
            print(f"\n{result}")
            return 0
        
        if args.ask_question:
            result = searcher.answer_question(args.ask_question)
            print(f"\n{result}")
            return 0
        
        # Standalone text operations
        if args.summarize_text:
            result = searcher.summarize_text(args.summarize_text)
            print(f"\n{result}")
            return 0
        
        if args.rewrite_text:
            result = searcher.rewrite_text(args.rewrite_text)
            print(f"\n{result}")
            return 0
        
        if args.improve_text:
            result = searcher.improve_text(args.improve_text)
            print(f"\n{result}")
            return 0
        
        if args.answer_question:
            if not args.context:
                print("❌ --context is required for question answering")
                return 1
            result = searcher.answer_with_context(args.answer_question, args.context)
            print(f"\n{result}")
            return 0
        
        if args.custom_prompt:
            result = searcher.process_custom_prompt(args.custom_prompt, args.context)
            print(f"\n{result}")
            return 0
    
    # Handle direct search query
    if args.query:
        # Build filters
        filters = {}
        if args.space:
            filters['space'] = args.space
        if args.creator:
            filters['creator'] = args.creator
        if args.date:
            filters['date'] = args.date
        
        # Perform search
        if args.search_type == 'semantic':
            results = searcher.semantic_search(args.query, args.top_k)
        elif args.search_type == 'boolean':
            results = searcher.boolean_search(args.query, args.top_k)
        else:  # smart
            results = searcher.smart_search(args.query, args.top_k, filters)
        
        if not results:
            print("❌ No results found.")
            return 0
        
        print(f"\n✅ Found {len(results)} result(s):\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   ID: {result.id}")
            print(f"   Match Type: {result.match_type.title()}")
            if result.score is not None:
                print(f"   Score: {result.score:.3f}")
            if result.space:
                print(f"   Space: {result.space}")
            if result.creator:
                print(f"   Creator: {result.creator}")
            if result.last_modified:
                print(f"   Last Modified: {result.last_modified}")
            print(f"   → {result.snippet}")
            if result.attachments:
                print(f"   📎 Attachments: {', '.join(result.attachments)}")
            print(f"   Explanation: {result.explanation}")
            print()
        
        return 0
    
    # Default to interactive mode
    enhanced_interactive_search(searcher)
    return 0

if __name__ == "__main__":
    exit(main())
