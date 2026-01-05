"""
Prometheus metrics collection for ComfyUI.
Collects and exports performance metrics in Prometheus format.
Python 3.14 & CUDA 13.x compatible.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Prometheus client import with graceful fallback
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, REGISTRY,
        start_http_server, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client loaded successfully")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("Prometheus client not available")
    
    # Create no-op classes for when Prometheus is unavailable
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1, labels=None): pass
        def labels(self, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value, labels=None): pass
        def time(self):
            class NoOpTimer:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return NoOpTimer()
        def labels(self, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def labels(self, **kwargs):
            return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
        def labels(self, **kwargs):
            return self


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time."""
    timestamp: str
    metric_name: str
    metric_type: str
    value: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class MetricsCollector:
    """Collects Prometheus metrics for Numba operations."""
    
    def __init__(self, registry=None):
        self.registry = registry or REGISTRY
        self.enabled = PROMETHEUS_AVAILABLE
        self._metrics_history: List[MetricSnapshot] = []
        
        if self.enabled:
            self._initialize_metrics()
        else:
            logger.warning("Prometheus not available, metrics disabled")
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        try:
            # Execution metrics
            self.numba_calls = Counter(
                'numba_function_calls_total',
                'Total number of Numba function calls',
                ['operation', 'function', 'implementation'],
                registry=self.registry
            )
            
            self.numba_errors = Counter(
                'numba_errors_total',
                'Total number of Numba execution errors',
                ['operation', 'function', 'error_type'],
                registry=self.registry
            )
            
            self.numba_fallbacks = Counter(
                'numba_fallbacks_total',
                'Total fallbacks to NumPy',
                ['operation', 'function', 'reason'],
                registry=self.registry
            )
            
            # Performance metrics
            self.execution_time = Histogram(
                'numba_execution_time_seconds',
                'Numba function execution time in seconds',
                ['operation', 'function'],
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
                registry=self.registry
            )
            
            self.memory_usage = Gauge(
                'numba_memory_usage_bytes',
                'Memory usage of Numba operations',
                ['operation', 'function'],
                registry=self.registry
            )
            
            self.array_size = Summary(
                'numba_array_size_elements',
                'Size of arrays processed',
                ['operation', 'function'],
                registry=self.registry
            )
            
            # GPU metrics (if available)
            self.gpu_memory = Gauge(
                'numba_gpu_memory_used_bytes',
                'GPU memory used by Numba operations',
                ['gpu_id', 'operation'],
                registry=self.registry
            )
            
            self.gpu_utilization = Gauge(
                'numba_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id'],
                registry=self.registry
            )
            
            # Cache metrics
            self.cache_hits = Counter(
                'numba_cache_hits_total',
                'Numba JIT cache hits',
                ['function'],
                registry=self.registry
            )
            
            self.cache_misses = Counter(
                'numba_cache_misses_total',
                'Numba JIT cache misses',
                ['function'],
                registry=self.registry
            )
            
            logger.debug("Prometheus metrics initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            self.enabled = False
    
    def record_execution(
        self,
        operation: str,
        function: str,
        execution_time: float,
        array_size: Optional[int] = None,
        memory_bytes: Optional[int] = None,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """Record a function execution."""
        if not self.enabled:
            return
        
        try:
            # Record call
            self.numba_calls.labels(
                operation=operation,
                function=function,
                implementation='numba'
            ).inc()
            
            # Record execution time
            self.execution_time.labels(
                operation=operation,
                function=function
            ).observe(execution_time)
            
            # Record array size if provided
            if array_size is not None:
                self.array_size.labels(
                    operation=operation,
                    function=function
                ).observe(array_size)
            
            # Record memory if provided
            if memory_bytes is not None:
                self.memory_usage.labels(
                    operation=operation,
                    function=function
                ).set(memory_bytes)
            
            # Record error if occurred
            if not success and error_type:
                self.numba_errors.labels(
                    operation=operation,
                    function=function,
                    error_type=error_type
                ).inc()
            
            # Store in history
            self._add_to_history(
                metric_name=function,
                metric_type='execution',
                value=execution_time,
                labels={'operation': operation}
            )
        
        except Exception as e:
            logger.debug(f"Failed to record execution metric: {e}")
    
    def record_fallback(
        self,
        operation: str,
        function: str,
        reason: str = "unknown"
    ):
        """Record a fallback to NumPy."""
        if not self.enabled:
            return
        
        try:
            self.numba_fallbacks.labels(
                operation=operation,
                function=function,
                reason=reason
            ).inc()
            
            self._add_to_history(
                metric_name=f"{function}_fallback",
                metric_type='fallback',
                value=1.0,
                labels={'operation': operation, 'reason': reason}
            )
        
        except Exception as e:
            logger.debug(f"Failed to record fallback metric: {e}")
    
    def record_cache_hit(self, function: str):
        """Record a JIT cache hit."""
        if not self.enabled or not self.cache_hits:
            return
        
        try:
            self.cache_hits.labels(function=function).inc()
        except Exception as e:
            logger.debug(f"Failed to record cache hit: {e}")
    
    def record_cache_miss(self, function: str):
        """Record a JIT cache miss."""
        if not self.enabled or not self.cache_misses:
            return
        
        try:
            self.cache_misses.labels(function=function).inc()
        except Exception as e:
            logger.debug(f"Failed to record cache miss: {e}")
    
    def record_gpu_memory(self, gpu_id: int, memory_bytes: int):
        """Record GPU memory usage."""
        if not self.enabled or not self.gpu_memory:
            return
        
        try:
            self.gpu_memory.labels(
                gpu_id=str(gpu_id),
                operation='numba'
            ).set(memory_bytes)
        except Exception as e:
            logger.debug(f"Failed to record GPU memory: {e}")
    
    def record_gpu_utilization(self, gpu_id: int, utilization_percent: float):
        """Record GPU utilization."""
        if not self.enabled or not self.gpu_utilization:
            return
        
        try:
            self.gpu_utilization.labels(
                gpu_id=str(gpu_id)
            ).set(utilization_percent)
        except Exception as e:
            logger.debug(f"Failed to record GPU utilization: {e}")
    
    def _add_to_history(
        self,
        metric_name: str,
        metric_type: str,
        value: float,
        labels: Dict[str, str]
    ):
        """Add metric to history for logging."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            labels=labels
        )
        self._metrics_history.append(snapshot)
        
        # Keep only last 1000 metrics
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]
    
    def get_metrics_history(
        self,
        metric_type: Optional[str] = None,
        limit: int = 100
    ) -> List[MetricSnapshot]:
        """Get recent metrics history."""
        history = self._metrics_history
        
        if metric_type:
            history = [m for m in history if m.metric_type == metric_type]
        
        return history[-limit:]
    
    def get_prometheus_output(self) -> bytes:
        """Get Prometheus metrics in text format."""
        if not self.enabled:
            return b""
        
        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error(f"Failed to generate Prometheus output: {e}")
            return b""
    
    def start_http_server(self, port: int = 8000):
        """Start HTTP server for Prometheus metrics."""
        if not self.enabled:
            logger.warning("Cannot start metrics server: Prometheus disabled")
            return
        
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


# Global metrics collector instance
_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance."""
    global _collector_instance
    
    if _collector_instance is None:
        _collector_instance = MetricsCollector()
    
    return _collector_instance


def initialize_metrics(registry=None) -> MetricsCollector:
    """Initialize the metrics collector."""
    global _collector_instance
    _collector_instance = MetricsCollector(registry)
    return _collector_instance
