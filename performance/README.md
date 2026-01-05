# Performance Metrics Storage

This directory contains timestamped performance metrics for ComfyUI Numba operations.

## Directory Structure

```
performance/
├── YYYY-MM-DD/                   # Daily folders (auto-created)
│   ├── session_[timestamp].log  # Session initialization info
│   ├── operations/               # Metrics by operation type
│   │   ├── image_processing_[timestamp].jsonl
│   │   ├── matrix_ops_[timestamp].jsonl
│   │   └── ...
│   ├── functions/                # Metrics by function name
│   │   ├── normalize_image_[timestamp].jsonl
│   │   ├── alpha_composite_[timestamp].jsonl
│   │   └── ...
│   ├── errors/                   # Error logs
│   │   └── errors_[timestamp].jsonl
│   ├── summaries/                # Daily summaries (auto-generated)
│   │   ├── operations_summary.json
│   │   ├── functions_summary.json
│   │   └── performance_summary.json
│   └── metrics_[timestamp].{csv,json}  # Full exports
└── archive/                      # Compressed/archived metrics
```

## Metric Format (JSONL)

Each metric is recorded as JSON Lines:

```json
{
  "timestamp": "2026-01-05T14:30:22.123456",
  "operation": "image_processing",
  "function": "normalize_image",
  "execution_time_ms": 12.345,
  "array_size": 1024000,
  "memory_bytes": 4194304,
  "success": true,
  "error_type": null,
  "fallback": false,
  "metadata": {}
}
```

## Quick Start

Performance metrics are automatically logged when Numba operations are traced. Access them:

```python
from comfy.performance_logger import get_performance_logger

logger = get_performance_logger()

# Get statistics
op_stats = logger.get_operation_stats("image_processing")
session_summary = logger.get_session_summary()

# Export metrics
logger.export_to_csv("metrics.csv")
logger.export_to_json("metrics.json")
```

## Integration

- OpenTelemetry tracing: http://localhost:16686 (Jaeger UI)
- Prometheus metrics: http://localhost:8000/metrics
- Performance logs: `performance/YYYY-MM-DD/*.jsonl`

See [OBSERVABILITY_GUIDE.md](../OBSERVABILITY_GUIDE.md) for complete documentation.
