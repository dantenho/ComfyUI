"""
OpenTelemetry instrumentation for ComfyUI Numba operations.
Provides distributed tracing and performance monitoring.
Python 3.14 & CUDA 13.x compatible.
"""

import logging
from typing import Optional, Any, Dict
from functools import wraps
import time

logger = logging.getLogger(__name__)

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
    
    OPENTELEMETRY_AVAILABLE = True
    logger.info("OpenTelemetry loaded successfully")
except ImportError as e:
    OPENTELEMETRY_AVAILABLE = False
    logger.debug(f"OpenTelemetry not available: {e}")


class TracingConfig:
    """Configuration for OpenTelemetry tracing."""
    
    def __init__(
        self,
        service_name: str = "comfyui",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        enabled: bool = True,
        sample_rate: float = 1.0
    ):
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.enabled = enabled and OPENTELEMETRY_AVAILABLE
        self.sample_rate = sample_rate
        self.tracer = None
        self.meter = None
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize OpenTelemetry components."""
        try:
            # Initialize tracer
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.jaeger_host,
                agent_port=self.jaeger_port,
            )
            
            trace_provider = TracerProvider()
            trace_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            trace.set_tracer_provider(trace_provider)
            self.tracer = trace.get_tracer(__name__)
            
            # Initialize metrics with Prometheus
            prometheus_reader = PrometheusMetricReader()
            meter_provider = MeterProvider(metric_readers=[prometheus_reader])
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
            logger.info(
                f"OpenTelemetry initialized: "
                f"Jaeger={self.jaeger_host}:{self.jaeger_port}, "
                f"Prometheus enabled"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.enabled = False


class NumbaTracer:
    """Wrapper for tracing Numba operations."""
    
    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self.enabled = self.config.enabled
        self.tracer = self.config.tracer
        self.meter = self.config.meter
        
        # Create counters and histograms if available
        if self.enabled and self.meter:
            try:
                self.execution_counter = self.meter.create_counter(
                    name="numba_executions_total",
                    description="Total number of Numba function executions",
                    unit="1"
                )
                
                self.error_counter = self.meter.create_counter(
                    name="numba_errors_total",
                    description="Total number of Numba execution errors",
                    unit="1"
                )
                
                self.execution_time_histogram = self.meter.create_histogram(
                    name="numba_execution_time_ms",
                    description="Numba function execution time",
                    unit="ms"
                )
                
                self.fallback_counter = self.meter.create_counter(
                    name="numba_fallbacks_total",
                    description="Total fallbacks to NumPy",
                    unit="1"
                )
                
                logger.debug("Prometheus metrics initialized")
            except Exception as e:
                logger.error(f"Failed to create metrics: {e}")
    
    def trace_operation(
        self,
        operation_name: str,
        function_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to trace Numba operations.
        
        Args:
            operation_name: Category of operation (e.g., "image_processing")
            function_name: Name of the function
            attributes: Additional attributes to attach to span
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled or not self.tracer:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(
                    f"{operation_name}.{function_name}"
                ) as span:
                    try:
                        # Set span attributes
                        span.set_attribute("operation_type", operation_name)
                        span.set_attribute("function_name", function_name)
                        span.set_attribute("implementation", "numba")
                        
                        if attributes:
                            for key, value in attributes.items():
                                if isinstance(value, (str, int, float, bool)):
                                    span.set_attribute(key, value)
                        
                        # Record metrics
                        if self.execution_counter:
                            self.execution_counter.add(
                                1,
                                {"operation": operation_name, "function": function_name}
                            )
                        
                        # Track execution time
                        start_time = time.time()
                        try:
                            result = func(*args, **kwargs)
                            execution_time = (time.time() - start_time) * 1000
                            
                            if self.execution_time_histogram:
                                self.execution_time_histogram.record(
                                    execution_time,
                                    {"operation": operation_name, "function": function_name}
                                )
                            
                            span.set_attribute("execution_time_ms", execution_time)
                            span.set_attribute("status", "success")
                            return result
                            
                        except Exception as e:
                            execution_time = (time.time() - start_time) * 1000
                            
                            if self.error_counter:
                                self.error_counter.add(
                                    1,
                                    {
                                        "operation": operation_name,
                                        "function": function_name,
                                        "error_type": type(e).__name__
                                    }
                                )
                            
                            span.set_attribute("execution_time_ms", execution_time)
                            span.set_attribute("status", "error")
                            span.set_attribute("error_type", type(e).__name__)
                            span.record_exception(e)
                            raise
                    
                    except Exception as e:
                        logger.error(f"Tracing error in {function_name}: {e}")
                        # Continue without tracing
                        return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def record_fallback(
        self,
        operation_name: str,
        function_name: str,
        reason: str = "unknown"
    ):
        """Record a fallback to NumPy."""
        if self.enabled and self.fallback_counter:
            try:
                self.fallback_counter.add(
                    1,
                    {
                        "operation": operation_name,
                        "function": function_name,
                        "reason": reason
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to record fallback: {e}")
    
    def start_span(self, span_name: str, attributes: Optional[Dict] = None):
        """
        Create and return a new span context.
        
        Usage:
            with tracer.start_span("operation") as span:
                # Do work
                span.set_attribute("key", value)
        """
        if not self.enabled or not self.tracer:
            # Return a no-op context manager
            class NoOpSpan:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def set_attribute(self, key, value):
                    pass
            return NoOpSpan()
        
        span_context = self.tracer.start_as_current_span(span_name)
        if attributes:
            span = span_context.__enter__()
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
            return span_context
        return span_context


# Global tracer instance
_tracer_instance: Optional[NumbaTracer] = None


def get_tracer(config: Optional[TracingConfig] = None) -> NumbaTracer:
    """Get or create the global tracer instance."""
    global _tracer_instance
    
    if _tracer_instance is None:
        _tracer_instance = NumbaTracer(config)
    
    return _tracer_instance


def initialize_tracing(
    service_name: str = "comfyui",
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    enabled: bool = True
) -> NumbaTracer:
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Service name for tracing
        jaeger_host: Jaeger agent host
        jaeger_port: Jaeger agent port
        enabled: Enable/disable tracing
    
    Returns:
        Configured tracer instance
    """
    config = TracingConfig(
        service_name=service_name,
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
        enabled=enabled
    )
    
    global _tracer_instance
    _tracer_instance = NumbaTracer(config)
    
    return _tracer_instance


# Convenience decorators
def trace_numba_operation(operation_name: str, **kwargs):
    """
    Decorator for tracing Numba operations.
    
    Usage:
        @trace_numba_operation("image_processing")
        def my_numba_function(arr):
            pass
    """
    tracer = get_tracer()
    
    def decorator(func):
        return tracer.trace_operation(
            operation_name=operation_name,
            function_name=func.__name__,
            attributes=kwargs
        )(func)
    
    return decorator
