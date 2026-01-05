"""
ComfyUI Tracing Integration Script
Initializes all observability components at startup
"""

import sys
import logging
from pathlib import Path

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


def setup_tracing():
    """Initialize all tracing and observability components."""
    
    print("\n" + "="*70)
    print("ComfyUI Observability Initialization")
    print("="*70 + "\n")
    
    # Initialize tracing
    print("📊 Initializing OpenTelemetry Tracing...")
    try:
        from comfy.tracing import initialize_tracing, OPENTELEMETRY_AVAILABLE
        
        if OPENTELEMETRY_AVAILABLE:
            tracer = initialize_tracing(
                service_name="comfyui",
                jaeger_host="localhost",
                jaeger_port=6831,
                enabled=True
            )
            print("   ✓ OpenTelemetry Tracing initialized")
            print("   → Jaeger UI: http://localhost:16686")
        else:
            print("   ⚠ OpenTelemetry not installed")
            print("   → Install: uv pip install -r requirements-observability.txt")
    except Exception as e:
        logger.warning(f"Tracing initialization failed: {e}")
        print(f"   ⚠ {e}")
    
    # Initialize metrics
    print("\n📈 Initializing Prometheus Metrics...")
    try:
        from comfy.metrics import initialize_metrics, PROMETHEUS_AVAILABLE
        
        if PROMETHEUS_AVAILABLE:
            metrics = initialize_metrics()
            metrics.start_http_server(port=8000)
            print("   ✓ Prometheus Metrics initialized")
            print("   → Metrics Endpoint: http://localhost:8000/metrics")
        else:
            print("   ⚠ Prometheus client not installed")
            print("   → Install: uv pip install -r requirements-observability.txt")
    except Exception as e:
        logger.warning(f"Metrics initialization failed: {e}")
        print(f"   ⚠ {e}")
    
    # Initialize performance logging
    print("\n📝 Initializing Performance Logging...")
    try:
        from comfy.performance_logger import initialize_performance_logger
        
        perf_logger = initialize_performance_logger(base_path="performance")
        print("   ✓ Performance Logging initialized")
        print(f"   → Logs: performance/{perf_logger.session_start.strftime('%Y-%m-%d')}/")
        print(f"   → Session ID: {perf_logger.session_id}")
    except Exception as e:
        logger.warning(f"Performance logging initialization failed: {e}")
        print(f"   ⚠ {e}")
    
    print("\n" + "="*70)
    print("Observability Status")
    print("="*70)
    print("📊 Tracing:     Check http://localhost:16686")
    print("📈 Metrics:     Check http://localhost:8000/metrics")
    print("📝 Logs:        Check performance/YYYY-MM-DD/ folder")
    print("📚 Docs:        See OBSERVABILITY_GUIDE.md")
    print("="*70 + "\n")


def setup_error_handler_tracing():
    """Configure error handler with tracing."""
    print("🔧 Configuring Error Handler Tracing...")
    
    try:
        # The error handler will automatically use initialized tracer/metrics/logger
        from comfy import numba_error_handler
        print("   ✓ Error handler configured with tracing")
    except Exception as e:
        logger.warning(f"Error handler configuration failed: {e}")


if __name__ == "__main__":
    setup_tracing()
    setup_error_handler_tracing()
    
    print("\n✅ Tracing setup complete. You can now use @numba_safe_wrapper")
    print("   Example:")
    print("   @numba_safe_wrapper(")
    print("       function_name='my_op',")
    print("       operation_name='image_processing',")
    print("       enable_tracing=True,")
    print("       enable_metrics=True,")
    print("       enable_perf_logging=True")
    print("   )")
    print("   def my_numba_function(arr): pass\n")
