#!/usr/bin/env python3

"""
Enhanced Mistral 7B Service for macOS - Batch Processing Compatible
Standalone service for text processing with command-line prompt support
Optimized for Mac's unified memory architecture and large dataset processing
Updated for new intelligent indexing system
"""

import argparse
import json
import sys
from typing import Optional, List, Dict, Any
import logging
import os
import time
import threading
from contextlib import contextmanager
from pathlib import Path
import platform
import pickle
import numpy as np

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

# Mac-optimized CPU settings
if platform.system() == "Darwin":
    if platform.processor() == 'arm':
        # Apple Silicon (M1/M2/M3)
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
    else:
        # Intel Mac
        os.environ['OMP_NUM_THREADS'] = '12'
        os.environ['MKL_NUM_THREADS'] = '12'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception for cross-platform compatibility"""
    pass

@contextmanager
def timeout(duration):
    """Cross-platform timeout context manager using threading"""
    result = [None]
    exception = [None]
    
    def timeout_handler():
        time.sleep(duration)
        if result[0] is None:
            exception[0] = TimeoutError(f"Operation timed out after {duration} seconds")
    
    timer = threading.Timer(duration, timeout_handler)
    timer.start()
    
    try:
        yield
        result[0] = True
    except Exception as e:
        exception[0] = e
    finally:
        timer.cancel()
        if exception[0]:
            raise exception[0]

class EnhancedMistralService:
    """Enhanced Mistral 7B service optimized for macOS and batch processing"""
    
    def __init__(self, model_path: str = "/Users/anupam.singhdeo/Softwares/llm/models/mistral/mistral.q4.gguf",
                 cache_dir: str = None):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure all directories exist
        for directory in [LOGS_DIR, EMBEDDINGS_DIR, INDEXES_DIR, PARSED_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.device = self._detect_best_device()
        
        # Performance tracking
        self.total_requests = 0
        self.total_time = 0.0
        self.cache_hits = 0
        
        self._load_model()
    
    def _detect_best_device(self):
        """Detect best device for Mac (MPS for Apple Silicon, CPU for Intel)"""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
                return "mps"
            else:
                logger.info("Using CPU for computation")
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _load_model(self):
        """Load Mistral model with Mac-specific optimizations"""
        try:
            # Check if using GGUF model (llama-cpp-python)
            if self.model_path.endswith('.gguf'):
                from llama_cpp import Llama
                logger.info(f"Loading GGUF model: {self.model_path}")
                
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_threads=6,  # Optimized for Mac
                    verbose=False
                )
                logger.info("GGUF model loaded successfully")
                return
            
            # Fallback to transformers for HF models
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            logger.info("Loading Mistral 7B model with macOS optimizations...")
            
            # Mac-specific torch settings
            if self.device == "mps":
                torch.set_num_threads(8)
            else:
                torch.set_num_threads(12)
            torch.set_num_interop_threads(2)
            
            # Check if it's a local path or HF model ID
            model_path = Path(self.model_path)
            is_local_path = model_path.exists() and model_path.is_dir()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=is_local_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Mac-optimized model loading
            model_kwargs = {
                "local_files_only": is_local_path,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_cache": True,
            }
            
            # Device-specific optimizations
            if self.device == "mps":
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": None,
                })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": "cpu",
                })
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move to device if using MPS
            if self.device == "mps":
                self.model = self.model.to("mps")
            
            # Create pipeline
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": 50,
                "do_sample": False,
                "batch_size": 1,
                "model_kwargs": {
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
            }
            
            if self.device == "cpu":
                pipeline_kwargs["device"] = -1
            else:
                pipeline_kwargs["device"] = 0
            
            self.pipe = pipeline("text-generation", **pipeline_kwargs)
            logger.info(f"Mistral model loaded successfully on {self.device.upper()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def _get_cache_key(self, prompt: str, context: str = None, max_tokens: int = 256) -> str:
        """Generate cache key for prompt/context combination"""
        import hashlib
        content = f"{prompt}|{context or ''}|{max_tokens}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        cache_file = self.cache_dir / f"mistral_cache_{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                self.cache_hits += 1
                return cached_data['response']
            except:
                pass
        return None
    
    def _cache_response(self, cache_key: str, response: str, prompt: str, context: str = None):
        """Cache response for future use"""
        cache_file = self.cache_dir / f"mistral_cache_{cache_key}.json"
        try:
            cache_data = {
                'response': response,
                'prompt': prompt,
                'context': context,
                'timestamp': time.time(),
                'device': self.device
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def call_mistral_with_timeout(self, prompt: str, context: str = None,
                                max_tokens: int = 256, use_cache: bool = True) -> str:
        """Call Mistral model with timeout and caching"""
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, context, max_tokens)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.debug("Using cached response")
                return cached_response
        
        try:
            with timeout(90):
                response = self.call_mistral(prompt, context, max_tokens)
                
                # Cache successful response
                if use_cache and response and not response.startswith("❌"):
                    self._cache_response(cache_key, response, prompt, context)
                
                return response
                
        except TimeoutError:
            logger.warning("First attempt timed out, retrying with reduced parameters")
            try:
                with timeout(45):
                    return self.call_mistral(prompt, context, max_tokens // 2)
            except TimeoutError:
                return "❌ Query timed out. Please try a shorter prompt or context."
    
    def call_mistral(self, prompt: str, context: str = None, max_tokens: int = 256) -> str:
        """Call Mistral model with prompt and optional context - ENHANCED VERSION"""
        start_time = time.time()
        self.total_requests += 1
        
        # Truncate for Mac memory constraints
        if context and len(context) > 1200:
            context = context[:1200] + "..."
        if len(prompt) > 400:
            prompt = prompt[:400] + "..."
        
        try:
            if not self.model:
                return "Error: Model not loaded"
            
            # Handle GGUF models with IMPROVED parameters
            if self.model_path.endswith('.gguf'):
                # IMPROVED: Better prompt formatting for different text types
                if context:
                    if len(context.strip()) < 50:
                        # For very short text, be more explicit
                        full_prompt = f"[INST] {prompt} Please provide a brief summary of this text: \"{context}\" [/INST]"
                    else:
                        full_prompt = f"[INST] {prompt}\n\nContext:\n{context} [/INST]"
                else:
                    full_prompt = f"[INST] {prompt} [/INST]"
                
                # Limit total prompt length
                if len(full_prompt) > 1500:
                    full_prompt = full_prompt[:1500] + " [/INST]"
                
                print(f"DEBUG: Full prompt sent to model: '{full_prompt}'")  # Debug line
                
                result = self.model(
                    full_prompt,
                    max_tokens=max_tokens,
                    stop=["[INST]", "</s>", "[/INST]"],  # IMPROVED: Better stop tokens
                    echo=False,
                    temperature=0.7,  # ADDED: Some randomness for better generation
                    top_p=0.9,        # ADDED: Nucleus sampling
                    top_k=40,         # ADDED: Top-k sampling
                    repeat_penalty=1.1,  # ADDED: Prevent repetition
                    seed=-1           # ADDED: Random seed for variety
                )
                generated_text = result["choices"][0]["text"].strip()
                
                print(f"DEBUG: Raw generated text: '{generated_text}'")  # Debug line
                
            else:
                # Use transformers pipeline
                if context:
                    full_prompt = f"[INST] {prompt}\n\nContext:\n{context} [/INST]"
                else:
                    full_prompt = f"[INST] {prompt} [/INST]"
                
                # Limit total prompt length
                if len(full_prompt) > 1500:
                    full_prompt = full_prompt[:1500] + " [/INST]"
                
                result = self.pipe(
                    full_prompt,
                    max_new_tokens=max_tokens,
                    return_full_text=False,
                    clean_up_tokenization_spaces=True,
                    do_sample=True,  # CHANGED: Enable sampling
                    temperature=0.7,  # ADDED: Temperature
                    top_p=0.9        # ADDED: Nucleus sampling
                )
                generated_text = result[0]['generated_text']
            
            end_time = time.time()
            inference_time = end_time - start_time
            self.total_time += inference_time
            
            logger.info(f"Inference took {inference_time:.2f} seconds on {self.device.upper()}")
            
            # IMPROVED: Handle empty responses
            if not generated_text or generated_text.isspace():
                return "The model generated an empty response. This might be due to the input being too short or the generation parameters needing adjustment."
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error with model: {e}"
    
    def batch_process_documents(self, documents: List[Dict[str, Any]],
                              task: str = "summarize",
                              batch_size: int = 8) -> List[Dict[str, Any]]:
        """Process multiple documents in batches"""
        logger.info(f"Batch processing {len(documents)} documents with task: {task}")
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = []
            
            for doc in batch:
                try:
                    if task == "summarize":
                        result = self.summarize_text(
                            doc.get('content', ''),
                            doc.get('title', '')
                        )
                    elif task == "improve":
                        result = self.improve_text(
                            doc.get('content', ''),
                            doc.get('title', '')
                        )
                    elif task == "rewrite":
                        result = self.rewrite_text(doc.get('content', ''))
                    else:
                        result = f"Unknown task: {task}"
                    
                    batch_results.append({
                        'doc_id': doc.get('id', 'unknown'),
                        'title': doc.get('title', ''),
                        'original_content': doc.get('content', ''),
                        'processed_content': result,
                        'task': task,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc.get('id', 'unknown')}: {e}")
                    batch_results.append({
                        'doc_id': doc.get('id', 'unknown'),
                        'error': str(e),
                        'task': task
                    })
            
            results.extend(batch_results)
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        return results
    
    def summarize_text(self, text: str, title: str = "") -> str:
        """Summarize given text with Mac optimizations"""
        if not text.strip():
            return "❌ No text provided to summarize."
        
        if len(text) > 1200:
            text = text[:1200] + "..."
        
        # IMPROVED: Better prompt for different text lengths
        if len(text.strip()) < 100:
            prompt = f"Please write a brief summary of this short text: \"{text}\""
        else:
            prompt = "Summarize this text in 2-3 sentences, focusing on key points:"
        
        print(f"DEBUG: Mistral service prompt = '{prompt}, text: {text}'")  # Debug line
        
        summary = self.call_mistral_with_timeout(prompt, text, max_tokens=120)
        
        title_part = f" of '{title}'" if title else ""
        return f"📄 **Summary{title_part}**\n\n{summary}"
    
    def rewrite_text(self, text: str, instruction: str = None) -> str:
        """Rewrite text for clarity and conciseness"""
        if not text.strip():
            return "❌ No text provided to rewrite."
        
        if len(text) > 1200:
            text = text[:1200] + "..."
        
        base_prompt = "Rewrite this text to be clearer and more concise:"
        if instruction:
            base_prompt = f"Rewrite this text with the following instruction: {instruction}"
        
        rewritten = self.call_mistral_with_timeout(base_prompt, text, max_tokens=150)
        return f"✏️ **Rewritten Text:**\n\n{rewritten}"
    
    def improve_text(self, text: str, section_name: str = "") -> str:
        """Improve text structure and clarity"""
        if not text.strip():
            return "❌ No text provided to improve."
        
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        prompt = "Improve this text by enhancing structure, clarity, and readability:"
        improved = self.call_mistral_with_timeout(prompt, text, max_tokens=200)
        
        section_part = f" ({section_name})" if section_name else ""
        return f"🔧 **Improved Section{section_part}:**\n\n{improved}"
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question using provided context"""
        if not context.strip():
            return "❌ No context provided to answer the question."
        
        if len(context) > 1000:
            context = context[:1000] + "..."
        
        prompt = f"Answer this question based on the provided context: {question}"
        answer = self.call_mistral_with_timeout(prompt, context, max_tokens=150)
        
        return f"💬 **Answer:**\n\n{answer}"
    
    def process_custom_prompt(self, prompt: str, context: str = None) -> str:
        """Process any custom prompt with optional context"""
        if not prompt.strip():
            return "❌ No prompt provided."
        
        result = self.call_mistral_with_timeout(prompt, context, max_tokens=200)
        return f"🤖 **Mistral Response:**\n\n{result}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0
        return {
            'total_requests': self.total_requests,
            'total_time': round(self.total_time, 2),
            'average_time_per_request': round(avg_time, 2),
            'cache_hits': self.cache_hits,
            'cache_hit_rate': round(self.cache_hits / self.total_requests * 100, 1) if self.total_requests > 0 else 0,
            'device': self.device.upper()
        }
    
    def save_batch_results(self, results: List[Dict[str, Any]],
                          output_file: str = None) -> str:
        """Save batch processing results"""
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"mistral_batch_results_{timestamp}.json"
        
        output_path = PARSED_DIR / output_file
        
        # Add performance stats to results
        batch_summary = {
            'results': results,
            'summary': {
                'total_documents': len(results),
                'successful_processes': len([r for r in results if 'error' not in r]),
                'failed_processes': len([r for r in results if 'error' in r]),
                'timestamp': time.time()
            },
            'performance_stats': self.get_performance_stats()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch results saved to {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Mistral 7B Service for macOS')
    
    parser.add_argument('--prompt', help='Direct prompt to process')
    parser.add_argument('--context', help='Context for the prompt')
    parser.add_argument('--task', choices=['summarize', 'rewrite', 'improve', 'answer', 'custom', 'batch'],
                       default='custom', help='Task type')
    parser.add_argument('--text', help='Text to process')
    parser.add_argument('--question', help='Question for Q&A tasks')
    parser.add_argument('--title', help='Title for summarization')
    parser.add_argument('--instruction', help='Custom instruction for rewriting')
    parser.add_argument('--section', help='Section name for improvement')
    parser.add_argument('--model-path', default='/Users/anupam.singhdeo/Softwares/llm/models/mistral/mistral.q4.gguf',
                       help='Path to GGUF model file or Hugging Face model ID')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--cache-dir', default=str(CACHE_DIR),
                       help='Cache directory')
    parser.add_argument('--batch-file', help='JSON file with documents for batch processing')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--output-file', help='Output file for batch results')
    parser.add_argument('--stats', action='store_true',
                       help='Show performance statistics')
    
    args = parser.parse_args()
    
    # Ensure base directories exist
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Initialize service
    service = EnhancedMistralService(args.model_path, args.cache_dir)
    
    # Handle batch processing
    if args.task == 'batch' and args.batch_file:
        try:
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Determine batch task
            batch_task = 'summarize'  # default
            if args.instruction:
                batch_task = 'rewrite'
            elif args.section:
                batch_task = 'improve'
            
            results = service.batch_process_documents(
                documents,
                task=batch_task,
                batch_size=args.batch_size
            )
            
            output_file = service.save_batch_results(results, args.output_file)
            print(f"✅ Batch processing complete! Results saved to: {output_file}")
            
            # Show stats
            stats = service.get_performance_stats()
            print(f"\n📊 Performance Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return 1
    
    # Handle single prompt processing
    if args.prompt:
        if args.task == 'summarize':
            text = args.text or args.prompt
            result = service.summarize_text(text, args.title or "")
        elif args.task == 'rewrite':
            text = args.text or args.prompt
            result = service.rewrite_text(text, args.instruction)
        elif args.task == 'improve':
            text = args.text or args.prompt
            result = service.improve_text(text, args.section or "")
        elif args.task == 'answer':
            question = args.question or args.prompt
            if not args.context:
                print("❌ --context is required for Q&A tasks")
                return 1
            result = service.answer_question(question, args.context)
        else:  # custom
            result = service.process_custom_prompt(args.prompt, args.context)
        
        print(result)
        
        if args.stats:
            stats = service.get_performance_stats()
            print(f"\n📊 Performance Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return 0
    
    # Interactive mode would go here
    print("🤖 Enhanced Mistral Service Ready")
    print("Use --prompt for direct processing or --batch-file for batch processing")
    return 0

if __name__ == "__main__":
    exit(main())
