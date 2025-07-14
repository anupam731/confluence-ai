## Prerequisite -- Python 3.8 or higher, Mistral 7B model 

## 1. Create and activate a virtual environment in the root of the project
python3 -m venv my_ml_env
source my_ml_env/bin/activate

## 2. Upgrade pip and core build tools
pip install --upgrade pip setuptools wheel

## 3. Install foundational ML libraries (PyTorch with MPS)
pip install torch torchvision torchaudio

## 4. Install other common libraries
pip install beautifulsoup4 lxml numpy sentence-transformers faiss-cpu spacy rapidfuzz huggingface_hub setuptools-rust transformers accelerate

## 5. Optional: If you need quantization for LLMs on Mac, consider llama-cpp-python:
pip install llama-cpp-python

## 6. Optional: Download spaCy models if needed
python -m spacy download en_core_web_sm

## 7. Optional: Verify PyTorch MPS (run within Python interpreter or a script)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"


project_root/
├── scripts/
│   ├── parser.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── searcher_mistral_mac.py
│   └── mistral_service_mac.py
├── data/
│   ├── raw/               ## original entities.xml or zipped dump
│   ├── parsed/            ## JSONL or JSON outputs (from parser)
│   ├── embeddings/        ## .npy files (from embedder)
│   └── indexes/           ## FAISS index files (from indexer)
├── logs/
│   └── processing.log
└── venv/


###### ATTEMPTS
## Script is created to switch to iterParse to handle larger files that can't be directly loaded to memory
## BATCHING is enabled to handle larger files
## batch-wise streaming embedding and write .npy in chunks
## HNSW for speed & memory efficiency: faiss.IndexHNSWFlat (small files)
## quantized model (like q4 GGUF) with llama.cpp wrapper

cd $project_root
############ PARSING THE XML FILE ########################
## Basic parsing with new structure
## Parse your XML file into batches
python scripts/parser.py data/raw/entities.xml \
  --batch-size 32 \
  --output-dir ./data/parsed \
  --output-prefix confluence_batch \
  2>&1 | tee logs/parsing.log



############ EMBEDDING THE JSON FILE ########################
## Process all batch files in parsed directory
python scripts/embedder.py

## Use specific model
python scripts/embedder.py --model all-mpnet-base-v2

## Custom chunk size for larger documents
python scripts/embedder.py --chunk-size 1024 --overlap 100

## Process specific batch files
python scripts/embedder.py --batch-files data/parsed/confluence_batch_0000.json data/parsed/confluence_batch_0001.json

## Use more processes for large datasets
python scripts/embedder.py --processes 12

## Custom output naming
python scripts/embedder.py --output-prefix large_dataset --job-id 20250712_production


############ INDEXING THE NPY FILE ########################
## Process all embedding files into single index
python scripts/indexer.py
python scripts/indexer.py --combine-files

## Force HNSW for small datasets
python scripts/indexer.py --force-hnsw


## Process specific embedding files
python scripts/indexer.py \
  --embeddings-file data/embeddings/confluence_embeddings_20250712.npy \
  --metadata-file data/embeddings/confluence_metadata_20250712.pkl

## Custom index name and type
python scripts/indexer.py --index-name production_v1 --index-type hnsw



############ RUNNING THE SEARCHER ########################
## Auto-discover and search
python scripts/searcher.py --query "kubernetes setup"

## Interactive mode
python scripts/searcher.py --interactive


## Summarize a page
python scripts/searcher.py --summarize-title "Kubernetes Setup Guide"

## Ask a question
python scripts/searcher.py --ask-question "How do I configure Jira with MySQL?"

## Improve text
python scripts/searcher.py --improve-text "This is some text to improve"


## List available indexes
python scripts/searcher.py --list-indexes

## Show system statistics
python scripts/searcher.py --stats

## summarize harry had a little lamb but the lamb had harry and they were lamb and man
## summarize: 321212804

############ PROJECT STRUCTURE ########################

project_root/
├── scripts/
│   ├── parser.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── searcher_mistral_mac.py
│   └── mistral_service_mac.py
├── data/
│   ├── raw/               ## original entities.xml
│   ├── parsed/            ## JSONL or JSON outputs (from parser)
│   ├── embeddings/        ## .npy files (from embedder)
│   └── indexes/           ## FAISS/HNSW index files (from indexer)
├── logs/
│   └── processing.log
└── venv/



