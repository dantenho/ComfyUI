# Observability Initialization Script
# Initialize all observability components at startup

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def initialize_observability(
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_perf_logging: bool = True,
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    metrics_port: int = 8000,
    perf_log_path: str = "performance"
):
    """
    Initialize all observability components.
    
    Args:
        enable_tracing: Enable OpenTelemetry tracing
        enable_metrics: Enable Prometheus metrics
        enable_perf_logging: Enable performance logging
        jaeger_host: Jaeger agent host
        jaeger_port: Jaeger agent port
        metrics_port: Prometheus metrics HTTP port
        perf_log_path: Performance logs directory
    """
    
    logger.info("=" * 60)
    logger.info("Initializing ComfyUI Observability")
    logger.info("=" * 60)
    
    # Initialize tracing
    if enable_tracing:
        try:
            from comfy.tracing import initialize_tracing
            tracer = initialize_tracing(
                service_name="comfyui",
                jaeger_host=jaeger_host,
                jaeger_port=jaeger_port,
                enabled=True
            )
            logger.info(f"✓ OpenTelemetry initialized (Jaeger: {jaeger_host}:{jaeger_port})")
        except ImportError:
            logger.warning("⚠ OpenTelemetry not available (install: pip install opentelemetry-api)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize tracing: {e}")
    
    # Initialize metrics
    if enable_metrics:
        try:
            from comfy.metrics import initialize_metrics
            metrics = initialize_metrics()
            
            # Start HTTP server
            metrics.start_http_server(metrics_port)
            logger.info(f"✓ Prometheus metrics initialized (http://localhost:{metrics_port}/metrics)")
        except ImportError:
            logger.warning("⚠ Prometheus client not available (install: pip install prometheus-client)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize metrics: {e}")
    
    # Initialize performance logging
    if enable_perf_logging:
        try:
            from comfy.performance_logger import initialize_performance_logger
            perf_logger = initialize_performance_logger(base_path=perf_log_path)
            logger.info(f"✓ Performance logging initialized ({perf_log_path})")
        except Exception as e:
            logger.error(f"✗ Failed to initialize performance logging: {e}")
    
    logger.info("=" * 60)
    logger.info("Observability Initialization Complete")
    logger.info("=" * 60)


# Configuration from environment variables
if __name__ == "__main__":
    enable_tracing = os.getenv("COMFYUI_ENABLE_TRACING", "1").lower() == "1"
    enable_metrics = os.getenv("COMFYUI_ENABLE_METRICS", "1").lower() == "1"
    enable_perf_logging = os.getenv("COMFYUI_ENABLE_PERF_LOGGING", "1").lower() == "1"
    
    jaeger_host = os.getenv("JAEGER_AGENT_HOST", "localhost")
    jaeger_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
    metrics_port = int(os.getenv("COMFYUI_METRICS_PORT", "8000"))
    perf_log_path = os.getenv("COMFYUI_PERF_LOG_PATH", "performance")
    
    initialize_observability(
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
        enable_perf_logging=enable_perf_logging,
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
        metrics_port=metrics_port,
        perf_log_path=perf_log_path
    )
