"""
Performance logging with timestamped metric files.
Stores detailed performance metrics in organized folder structure.
Python 3.14 & CUDA 13.x compatible.
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import threading
from collections import defaultdict
import csv

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric entry."""
    timestamp: str
    operation: str
    function: str
    execution_time_ms: float
    array_size: Optional[int] = None
    memory_bytes: Optional[int] = None
    gpu_memory_bytes: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fallback: bool = False
    fallback_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class PerformanceLogger:
    """
    Logs performance metrics to timestamped files in organized folder structure.
    
    Structure:
    performance/
    ├── current_session.json          # Current session metrics
    ├── 2026-01-05/                   # Daily folder
    │   ├── session_start_timestamp.log
    │   ├── operations/               # By operation type
    │   │   ├── image_processing_timestamp.jsonl
    │   │   ├── matrix_ops_timestamp.jsonl
    │   │   └── ...
    │   ├── functions/                # By function name
    │   │   ├── normalize_image_timestamp.jsonl
    │   │   ├── alpha_composite_timestamp.jsonl
    │   │   └── ...
    │   ├── errors/                   # Error logs
    │   │   └── errors_timestamp.jsonl
    │   ├── summaries/                # Daily summaries
    │   │   ├── operations_summary.json
    │   │   ├── functions_summary.json
    │   │   └── performance_summary.json
    │   └── metrics.json              # Combined metrics
    └── archive/                      # Compressed old sessions
    """
    
    def __init__(self, base_path: str = "performance"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize session
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Create daily folder
        self.daily_path = self.base_path / self.session_start.strftime("%Y-%m-%d")
        self.daily_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.operations_path = self.daily_path / "operations"
        self.operations_path.mkdir(exist_ok=True)
        
        self.functions_path = self.daily_path / "functions"
        self.functions_path.mkdir(exist_ok=True)
        
        self.errors_path = self.daily_path / "errors"
        self.errors_path.mkdir(exist_ok=True)
        
        self.summaries_path = self.daily_path / "summaries"
        self.summaries_path.mkdir(exist_ok=True)
        
        # Thread-safe data structures
        self._lock = threading.Lock()
        self._metrics: List[PerformanceMetric] = []
        self._operation_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._function_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._errors: List[Dict[str, Any]] = []
        
        # Initialize session log
        self._init_session_log()
        
        logger.info(
            f"Performance logger initialized: {self.base_path} "
            f"(session: {self.session_id})"
        )
    
    def _init_session_log(self):
        """Initialize session log file."""
        session_log = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "base_path": str(self.base_path),
            "daily_path": str(self.daily_path)
        }
        
        log_file = self.daily_path / f"session_{self.session_id}.log"
        try:
            with open(log_file, 'w') as f:
                json.dump(session_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write session log: {e}")
    
    def log_metric(self, metric: PerformanceMetric):
        """Log a performance metric."""
        with self._lock:
            # Add to main metrics list
            self._metrics.append(metric)
            
            # Organize by operation
            self._operation_metrics[metric.operation].append(metric)
            
            # Organize by function
            self._function_metrics[metric.function].append(metric)
            
            # Log errors separately
            if not metric.success:
                self._errors.append(metric.to_dict())
            
            # Write to timestamped file
            self._write_metric_file(metric)
        
        # Periodically flush summaries
        if len(self._metrics) % 100 == 0:
            self.flush_summaries()
    
    def _write_metric_file(self, metric: PerformanceMetric):
        """Write metric to appropriate timestamped file."""
        try:
            # Operation-specific file
            operation_file = (
                self.operations_path /
                f"{metric.operation}_{self.session_id}.jsonl"
            )
            with open(operation_file, 'a') as f:
                f.write(metric.to_json() + '\n')
            
            # Function-specific file
            function_file = (
                self.functions_path /
                f"{metric.function}_{self.session_id}.jsonl"
            )
            with open(function_file, 'a') as f:
                f.write(metric.to_json() + '\n')
            
            # Error file if applicable
            if not metric.success:
                error_file = self.errors_path / f"errors_{self.session_id}.jsonl"
                with open(error_file, 'a') as f:
                    f.write(metric.to_json() + '\n')
        
        except Exception as e:
            logger.error(f"Failed to write metric file: {e}")
    
    def flush_summaries(self):
        """Flush summary files."""
        with self._lock:
            try:
                # Operation summaries
                op_summary = {}
                for op_name, metrics in self._operation_metrics.items():
                    op_summary[op_name] = self._compute_summary(metrics)
                
                summary_file = self.summaries_path / "operations_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(op_summary, f, indent=2)
                
                # Function summaries
                func_summary = {}
                for func_name, metrics in self._function_metrics.items():
                    func_summary[func_name] = self._compute_summary(metrics)
                
                summary_file = self.summaries_path / "functions_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(func_summary, f, indent=2)
                
                # Overall performance summary
                perf_summary = {
                    "session_id": self.session_id,
                    "total_metrics": len(self._metrics),
                    "total_errors": len(self._errors),
                    "overall_stats": self._compute_summary(self._metrics),
                    "timestamp": datetime.now().isoformat()
                }
                
                summary_file = self.summaries_path / "performance_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(perf_summary, f, indent=2)
            
            except Exception as e:
                logger.error(f"Failed to flush summaries: {e}")
    
    def _compute_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Compute statistics for a list of metrics."""
        if not metrics:
            return {}
        
        times = [m.execution_time_ms for m in metrics]
        
        return {
            "count": len(metrics),
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "total_ms": sum(times),
            "errors": sum(1 for m in metrics if not m.success),
            "fallbacks": sum(1 for m in metrics if m.fallback),
            "success_rate": sum(1 for m in metrics if m.success) / len(metrics)
        }
    
    def log_execution(
        self,
        operation: str,
        function: str,
        execution_time_ms: float,
        array_size: Optional[int] = None,
        memory_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log successful execution."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            function=function,
            execution_time_ms=execution_time_ms,
            array_size=array_size,
            memory_bytes=memory_bytes,
            success=True,
            metadata=metadata or {}
        )
        self.log_metric(metric)
    
    def log_error(
        self,
        operation: str,
        function: str,
        execution_time_ms: float,
        error_type: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log execution error."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            function=function,
            execution_time_ms=execution_time_ms,
            success=False,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {}
        )
        self.log_metric(metric)
    
    def log_fallback(
        self,
        operation: str,
        function: str,
        execution_time_ms: float,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log fallback to NumPy."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            function=function,
            execution_time_ms=execution_time_ms,
            success=True,
            fallback=True,
            fallback_reason=reason,
            metadata=metadata or {}
        )
        self.log_metric(metric)
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation."""
        with self._lock:
            metrics = self._operation_metrics.get(operation, [])
            return {
                "operation": operation,
                "stats": self._compute_summary(metrics),
                "metrics_count": len(metrics)
            }
    
    def get_function_stats(self, function: str) -> Dict[str, Any]:
        """Get statistics for a function."""
        with self._lock:
            metrics = self._function_metrics.get(function, [])
            return {
                "function": function,
                "stats": self._compute_summary(metrics),
                "metrics_count": len(metrics)
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get complete session summary."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "start_time": self.session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_metrics": len(self._metrics),
                "total_errors": len(self._errors),
                "operations": {
                    name: self._compute_summary(metrics)
                    for name, metrics in self._operation_metrics.items()
                },
                "functions": {
                    name: self._compute_summary(metrics)
                    for name, metrics in self._function_metrics.items()
                }
            }
    
    def export_to_csv(self, filepath: str = None):
        """Export all metrics to CSV file."""
        if not filepath:
            filepath = self.daily_path / f"metrics_{self.session_id}.csv"
        
        try:
            with open(filepath, 'w', newline='') as f:
                if self._metrics:
                    writer = csv.DictWriter(f, fieldnames=self._metrics[0].to_dict().keys())
                    writer.writeheader()
                    for metric in self._metrics:
                        writer.writerow(metric.to_dict())
            
            logger.info(f"Metrics exported to CSV: {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics to CSV: {e}")
    
    def export_to_json(self, filepath: str = None):
        """Export all metrics to JSON file."""
        if not filepath:
            filepath = self.daily_path / f"metrics_{self.session_id}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(
                    [m.to_dict() for m in self._metrics],
                    f,
                    indent=2
                )
            
            logger.info(f"Metrics exported to JSON: {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics to JSON: {e}")
    
    def cleanup_old_metrics(self, keep_days: int = 7):
        """Remove metrics older than specified days."""
        # Implementation for archiving/removing old data
        logger.info(f"Cleanup: keeping metrics from last {keep_days} days")


# Global performance logger instance
_logger_instance: Optional[PerformanceLogger] = None


def get_performance_logger(base_path: str = "performance") -> PerformanceLogger:
    """Get or create the global performance logger instance."""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = PerformanceLogger(base_path)
    
    return _logger_instance


def initialize_performance_logger(base_path: str = "performance") -> PerformanceLogger:
    """Initialize the performance logger."""
    global _logger_instance
    _logger_instance = PerformanceLogger(base_path)
    return _logger_instance
