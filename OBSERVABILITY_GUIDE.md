# OpenTelemetry & Prometheus Integration Guide

## Overview

ComfyUI now includes comprehensive observability with:
- **OpenTelemetry**: Distributed tracing for Numba operations
- **Prometheus**: Metrics collection and monitoring
- **Performance Logging**: Timestamped metric files organized by operation

## Installation

```bash
# Install optional observability dependencies
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-jaeger
pip install prometheus-client
```

## Components

### 1. OpenTelemetry Tracing (`comfy/tracing.py`)

Provides distributed tracing for all Numba operations.

**Features**:
- Automatic span creation for operations
- Performance metrics tracking
- Exception recording
- Jaeger backend support

**Usage**:

```python
from comfy.tracing import initialize_tracing, trace_numba_operation

# Initialize tracing (optional Jaeger backend)
tracer = initialize_tracing(
    service_name="comfyui",
    jaeger_host="localhost",
    jaeger_port=6831,
    enabled=True
)

# Use decorator
@trace_numba_operation("image_processing", detail_level="high")
def my_numba_function(arr):
    pass

# Or use context manager
with tracer.start_span("operation_name", attributes={"user_id": "123"}):
    result = numba_function(data)
```

### 2. Prometheus Metrics (`comfy/metrics.py`)

Collects and exports Prometheus metrics.

**Metrics Tracked**:
- `numba_function_calls_total`: Total function calls
- `numba_errors_total`: Error count by type
- `numba_fallbacks_total`: Fallback to NumPy count
- `numba_execution_time_seconds`: Function execution time (histogram)
- `numba_memory_usage_bytes`: Memory consumption
- `numba_array_size_elements`: Array sizes processed
- `numba_gpu_memory_used_bytes`: GPU memory (if available)
- `numba_gpu_utilization_percent`: GPU utilization
- `numba_cache_hits_total`: JIT cache hits
- `numba_cache_misses_total`: JIT cache misses

**Usage**:

```python
from comfy.metrics import initialize_metrics, get_metrics_collector

# Initialize Prometheus metrics
metrics = initialize_metrics()

# Metrics are automatically collected via decorator/error handler

# Start HTTP server for scraping
metrics.start_http_server(port=8000)
# Prometheus can now scrape: http://localhost:8000/metrics

# Get metrics history
history = metrics.get_metrics_history(metric_type='execution', limit=100)

# Generate Prometheus format output
prometheus_output = metrics.get_prometheus_output()
```

### 3. Performance Logging (`comfy/performance_logger.py`)

Logs detailed performance metrics to timestamped files.

**Directory Structure**:
```
performance/
├── current_session.json
├── 2026-01-05/                          # Daily folder
│   ├── session_20260105_143022.log     # Session metadata
│   ├── operations/                      # By operation type
│   │   ├── image_processing_[timestamp].jsonl
│   │   ├── matrix_ops_[timestamp].jsonl
│   │   └── ...
│   ├── functions/                       # By function name
│   │   ├── normalize_image_[timestamp].jsonl
│   │   ├── alpha_composite_[timestamp].jsonl
│   │   └── ...
│   ├── errors/                          # Error logs
│   │   └── errors_[timestamp].jsonl
│   ├── summaries/                       # Daily summaries
│   │   ├── operations_summary.json
│   │   ├── functions_summary.json
│   │   └── performance_summary.json
│   └── metrics_[timestamp].{csv,json}  # Full exports
└── archive/
```

**Usage**:

```python
from comfy.performance_logger import initialize_performance_logger

# Initialize with custom path
perf_logger = initialize_performance_logger(base_path="performance")

# Log successful execution
perf_logger.log_execution(
    operation="image_processing",
    function="normalize_image",
    execution_time_ms=12.5,
    array_size=1024000,
    metadata={"device": "cuda"}
)

# Log error
perf_logger.log_error(
    operation="image_processing",
    function="normalize_image",
    execution_time_ms=8.2,
    error_type="RuntimeError",
    error_message="CUDA out of memory"
)

# Log fallback
perf_logger.log_fallback(
    operation="image_processing",
    function="normalize_image",
    execution_time_ms=15.3,
    reason="CUDA unavailable"
)

# Get statistics
op_stats = perf_logger.get_operation_stats("image_processing")
func_stats = perf_logger.get_function_stats("normalize_image")
session_summary = perf_logger.get_session_summary()

# Export metrics
perf_logger.export_to_csv("metrics.csv")
perf_logger.export_to_json("metrics.json")

# Flush summaries periodically
perf_logger.flush_summaries()
```

## Integration with Error Handler

The error handler automatically integrates with observability:

```python
from comfy.numba_error_handler import numba_safe_wrapper

@numba_safe_wrapper(
    fallback_func=numpy_normalize,
    function_name="normalize_image",
    operation_name="image_processing",
    enable_tracing=True,      # Enable OpenTelemetry
    enable_metrics=True,       # Enable Prometheus
    enable_perf_logging=True,  # Enable timestamped logs
    silent=False
)
def numba_normalize(arr):
    # Numba optimized code
    pass
```

## Monitoring Setup

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'comfyui'
    static_configs:
      - targets: ['localhost:8000']
```

Start Prometheus:
```bash
prometheus --config.file=prometheus.yml
```

### Jaeger Setup

Start Jaeger all-in-one:
```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 6831:6831/udp \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Access Jaeger UI: http://localhost:16686

### Grafana Dashboards

Import Prometheus as datasource and create dashboards for:
- Numba execution time trends
- Error rates by function
- Fallback frequency
- GPU memory usage
- Cache hit rates

## Performance Analysis

### Query Prometheus Metrics

Examples using PromQL:

```promql
# Average execution time by operation
avg(rate(numba_execution_time_seconds[5m])) by (operation)

# Error rate by function
rate(numba_errors_total[5m]) / rate(numba_function_calls_total[5m])

# Fallback rate
rate(numba_fallbacks_total[5m])

# P95 latency
histogram_quantile(0.95, numba_execution_time_seconds)

# GPU memory trend
numba_gpu_memory_used_bytes
```

### Analyze Performance Logs

```python
import json
from pathlib import Path

# Read timestamped metrics
metrics_file = Path("performance/2026-01-05/operations/image_processing_*.jsonl")

metrics = []
for line in open(metrics_file).readlines():
    metrics.append(json.loads(line))

# Calculate statistics
times = [m['execution_time_ms'] for m in metrics]
errors = [m for m in metrics if not m['success']]
fallbacks = [m for m in metrics if m['fallback']]

print(f"Average time: {sum(times)/len(times):.2f}ms")
print(f"Error rate: {len(errors)/len(metrics)*100:.1f}%")
print(f"Fallback rate: {len(fallbacks)/len(metrics)*100:.1f}%")
```

## Logging Configuration

```python
import logging
import json
from pathlib import Path

# Configure logging with performance metadata
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Add performance logging handler
perf_handler = logging.FileHandler("performance.log")
perf_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
perf_handler.setFormatter(perf_formatter)

logger = logging.getLogger("comfy.numba_error_handler")
logger.addHandler(perf_handler)
```

## Environment Variables

```bash
# Enable/disable tracing
export COMFYUI_ENABLE_TRACING=1

# Jaeger configuration
export JAEGER_AGENT_HOST=localhost
export JAEGER_AGENT_PORT=6831

# Prometheus metrics server
export COMFYUI_METRICS_PORT=8000

# Performance logging
export COMFYUI_PERF_LOG_PATH=performance
export COMFYUI_PERF_LOG_LEVEL=INFO
```

## Best Practices

1. **Sampling**: Use sampling for high-volume operations
   ```python
   tracer = initialize_tracing(sample_rate=0.1)  # 10% sampling
   ```

2. **Performance**: Disable observability in performance-critical paths if needed
   ```python
   @numba_safe_wrapper(
       enable_tracing=False,
       enable_metrics=False,
       enable_perf_logging=False
   )
   ```

3. **Storage**: Periodically archive old performance logs
   ```python
   perf_logger.cleanup_old_metrics(keep_days=7)
   ```

4. **Monitoring**: Set up alerts for error rates and latency
   - High error rate: `rate(numba_errors_total[5m]) > 0.05`
   - High latency: `histogram_quantile(0.95, numba_execution_time_seconds) > 0.5`
   - Frequent fallbacks: `rate(numba_fallbacks_total[5m]) > 0.1`

## Troubleshooting

### OpenTelemetry not available
```
Warning: OpenTelemetry not available, tracing disabled
Solution: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger
```

### Prometheus metrics not updating
- Ensure Prometheus is started: `start_http_server(8000)`
- Check scrape interval in prometheus.yml
- Verify port 8000 is accessible

### Performance logs not created
- Check directory permissions: `ls -la performance/`
- Verify base_path is writable
- Check disk space

### High overhead
- Disable unused observability: `enable_tracing=False`
- Use sampling instead of 100% tracing
- Archive old metrics regularly

## Metrics Reference

See `PERFORMANCE_METRICS.md` for detailed metrics reference.
