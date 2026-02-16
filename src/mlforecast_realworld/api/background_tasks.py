"""
Background task management for long-running operations.

This module provides a task manager for handling data downloads
and model training in the background without blocking the API.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Type of background task."""
    DATA_UPDATE = "data_update"
    MODEL_TRAINING = "model_training"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class TaskInfo:
    """Information about a background task."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tickers_requested: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "tickers_requested": self.tickers_requested,
        }


class BackgroundTaskManager:
    """
    Manager for background tasks.

    Handles running long tasks (data download, model training) in the background
    while providing status updates to the frontend.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._current_task_id: str | None = None

    def create_task(
        self,
        task_type: TaskType,
        tickers: list[str] | None = None,
    ) -> TaskInfo:
        """Create a new task entry."""
        task_id = str(uuid.uuid4())[:8]
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            tickers_requested=tickers or [],
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tasks."""
        with self._lock:
            return list(self._tasks.values())

    def get_latest_task(self, task_type: TaskType | None = None) -> TaskInfo | None:
        """Get the most recent task, optionally filtered by type."""
        with self._lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            if not tasks:
                return None
            return max(tasks, key=lambda t: t.created_at)

    def get_current_task(self) -> TaskInfo | None:
        """Get the currently running task."""
        with self._lock:
            if self._current_task_id:
                return self._tasks.get(self._current_task_id)
            return None

    def is_busy(self) -> bool:
        """Check if any task is currently running."""
        with self._lock:
            return self._current_task_id is not None

    def update_task(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update task status."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if status:
                task.status = status
                if status == TaskStatus.RUNNING and task.started_at is None:
                    task.started_at = datetime.now()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    task.completed_at = datetime.now()
                    if self._current_task_id == task_id:
                        self._current_task_id = None
            if progress is not None:
                task.progress = progress
            if message is not None:
                task.message = message
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

    def run_in_background(
        self,
        task_id: str,
        func: Callable[[], dict[str, Any]],
    ) -> None:
        """Run a function in a background thread."""

        def _worker():
            try:
                with self._lock:
                    self._current_task_id = task_id
                self.update_task(task_id, status=TaskStatus.RUNNING, message="Starting...")
                result = func()
                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=100.0,
                    message="Completed successfully",
                    result=result,
                )
            except Exception as e:
                logger.exception("Background task %s failed", task_id)
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"Failed: {e}",
                    error=str(e),
                )
            finally:
                with self._lock:
                    if self._current_task_id == task_id:
                        self._current_task_id = None

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def clear_completed(self) -> int:
        """Remove completed/failed tasks. Returns count removed."""
        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            for tid in to_remove:
                del self._tasks[tid]
            return len(to_remove)


# Global instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager

