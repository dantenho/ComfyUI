# ComfyUI Documentation Index

All project documentation is stored in **ChromaDB** for easy semantic search and retrieval.

## Quick Access

```python
import chromadb

client = chromadb.Client()
```

## Available Collections

### 1. `comfyui_migration_guide` (10 docs)
Python 3.14 & CUDA 13.x migration information
```python
migration = client.get_collection('comfyui_migration_guide')
results = migration.query(
    query_texts=["How to update Python 3.14 async code"],
    n_results=5
)
```

### 2. `numba_pytorch_docs` (23 docs)
Numba & PyTorch API documentation and examples
```python
api_docs = client.get_collection('numba_pytorch_docs')

# Query Numba
numba_results = api_docs.query(
    query_texts=["Numba CUDA kernel programming"],
    where={"category": "numba"},
    n_results=5
)

# Query PyTorch
pytorch_results = api_docs.query(
    query_texts=["PyTorch torch.compile optimization"],
    where={"category": "pytorch"},
    n_results=5
)
```

### 3. `comfyui_docs` (5 docs)
Project documentation, CI/CD, configuration guides
```python
docs = client.get_collection('comfyui_docs')

# Find CI/CD info
cicd_results = docs.query(
    query_texts=["GitHub Actions workflow setup"],
    n_results=3
)

# Find Ruff config
ruff_results = docs.query(
    query_texts=["Ruff linter configuration"],
    n_results=2
)
```

## Common Queries

### Migration Help
```python
# Python 3.14 changes
results = migration.query(
    query_texts=["asyncio deprecated functions"],
    n_results=3
)

# CUDA 13.x updates
results = migration.query(
    query_texts=["CUDA memory management"],
    n_results=5
)
```

### API Documentation
```python
# Numba JIT compilation
results = api_docs.query(
    query_texts=["JIT decorator options"],
    where={"category": "numba", "topic": "jit"},
    n_results=5
)

# PyTorch new features
results = api_docs.query(
    query_texts=["symmetric memory API"],
    where={"category": "pytorch"},
    n_results=3
)
```

### CI/CD Setup
```python
# Workflow information
results = docs.query(
    query_texts=["GitHub Actions testing workflow"],
    where={"topic": "github_actions"},
    n_results=3
)

# Linting setup
results = docs.query(
    query_texts=["Ruff configuration"],
    where={"topic": "ruff_linter"},
    n_results=2
)
```

## Document Categories

| Collection | Documents | Topics |
|------------|-----------|---------|
| **comfyui_migration_guide** | 10 | Python 3.14, CUDA 13.x, breaking changes, APIs |
| **numba_pytorch_docs** | 23 | Numba, PyTorch, CUDA, JIT, torch.compile |
| **comfyui_docs** | 5 | Migration, CI/CD, Ruff, ChromaDB |

## List All Collections
```python
collections = client.list_collections()
for coll in collections:
    print(f"Collection: {coll.name}")
    print(f"Count: {coll.count()}")
    print()
```

## Browse Collection Contents
```python
# Get collection
coll = client.get_collection("comfyui_docs")

# Get sample documents
sample = coll.peek(limit=3)
print(sample['documents'])
print(sample['metadatas'])
```

## Search with Filters
```python
# By date
results = docs.query(
    query_texts=["documentation"],
    where={"date": "2026-01-05"},
    n_results=10
)

# By category
results = docs.query(
    query_texts=["configuration"],
    where={"category": "documentation"},
    n_results=5
)

# By type
results = migration.query(
    query_texts=["Python updates"],
    where={"type": "migration_guide"},
    n_results=5
)
```

## Why ChromaDB?

✅ **No File Clutter**: All docs in database, not scattered files  
✅ **Semantic Search**: Find docs by meaning, not just keywords  
✅ **Metadata Filtering**: Query by category, date, topic, type  
✅ **Always Updated**: Single source of truth  
✅ **Easy Access**: Simple Python API  

## Alternative: View in Web UI

If ChromaDB server is running with UI:
```bash
# Start ChromaDB server (if not already running)
chroma run --path ./chroma_db

# Access at: http://localhost:8000
```

---

**Last Updated**: 2026-01-05  
**Total Documents**: 38 (10 + 23 + 5)  
**Collections**: 3  
**Status**: ✅ All docs in ChromaDB
