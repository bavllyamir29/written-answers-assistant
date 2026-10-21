import hashlib
import hmac
import json
import logging
import logging.handlers
import os
import sqlite3
import time
import traceback
import uuid
from contextlib import closing
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from services.resilience import CircuitBreaker

P = ParamSpec("P")
R = TypeVar("R")

_REQUEST_ID: ContextVar[str | None] = ContextVar("grading_request_id", default=None)
_LOGGER_NAME = "grading_audit"
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_LOG_DIRECTORY = Path("logs")
_JSON_LOG_PATH = _LOG_DIRECTORY / "app.log"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _student_hash(student_id: Any) -> str | None:
    if student_id is None:
        return None
    value = str(student_id).encode("utf-8")
    salt = os.getenv("STUDENT_ID_HASH_SALT", "").encode("utf-8")
    if salt:
        return hmac.new(salt, value, hashlib.sha256).hexdigest()
    return hashlib.sha256(value).hexdigest()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = dict(record.audit_data)
        entry["level"] = record.levelname
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str)


class _SQLiteAuditHandler(logging.Handler):
    def __init__(self, database_path: Path) -> None:
        super().__init__()
        self._database_path = database_path
        self._breaker = CircuitBreaker(
            int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            float(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30")),
        )
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._database_path, timeout=0.2)) as connection:
                with connection:
                    connection.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS grading_logs (
                            request_id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            student_id_hash TEXT,
                            question_id TEXT,
                            student_answer TEXT,
                            model_answer TEXT,
                            ai_score REAL,
                            ai_feedback TEXT,
                            model_name TEXT,
                            tokens_used INTEGER,
                            estimated_cost REAL,
                            response_time_seconds REAL,
                            status TEXT NOT NULL,
                            error_message TEXT,
                            stack_trace TEXT
                        );
                        CREATE TABLE IF NOT EXISTS teacher_overrides (
                            override_id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            request_id TEXT NOT NULL,
                            student_id_hash TEXT,
                            question_id TEXT,
                            old_score REAL NOT NULL,
                            new_score REAL NOT NULL,
                            comment TEXT
                        );
                        """
                    )
        except Exception as exc:
            opened = self._breaker.record_failure()
            logging.getLogger("uvicorn.error").warning(
                "Fallback event dependency=grading_log_database error=%s "
                "outcome=continue_without_database circuit_opened=%s",
                exc,
                opened,
            )

    def emit(self, record: logging.LogRecord) -> None:
        entry = record.audit_data
        if entry.get("event_type") not in {"grading", "teacher_override"}:
            return
        if not self._breaker.allow_request():
            logging.getLogger("uvicorn.error").warning(
                "Fallback event dependency=grading_log_database "
                "outcome=circuit_open"
            )
            return
        try:
            if entry.get("event_type") == "grading":
                with closing(sqlite3.connect(self._database_path, timeout=0.2)) as connection:
                    with connection:
                        connection.execute(
                            """
                            INSERT INTO grading_logs (
                                request_id, timestamp, student_id_hash, question_id,
                                student_answer, model_answer, ai_score, ai_feedback,
                                model_name, tokens_used, estimated_cost,
                                response_time_seconds, status, error_message, stack_trace
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entry["request_id"],
                                entry["timestamp"],
                                entry["student_id_hash"],
                                entry["question_id"],
                                entry["student_answer"],
                                entry["model_answer"],
                                entry["ai_score"],
                                entry["ai_feedback"],
                                entry["model_name"],
                                entry["tokens_used"],
                                entry["estimated_cost"],
                                entry["response_time_seconds"],
                                entry["status"],
                                entry["error_message"],
                                entry["stack_trace"],
                            ),
                        )
            elif entry.get("event_type") == "teacher_override":
                with closing(sqlite3.connect(self._database_path, timeout=0.2)) as connection:
                    with connection:
                        connection.execute(
                            """
                            INSERT INTO teacher_overrides (
                                override_id, timestamp, request_id, student_id_hash,
                                question_id, old_score, new_score, comment
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entry["override_id"],
                                entry["timestamp"],
                                entry["request_id"],
                                entry["student_id_hash"],
                                entry["question_id"],
                                entry["old_score"],
                                entry["new_score"],
                                entry["comment"],
                            ),
                        )
        except Exception as exc:
            opened = self._breaker.record_failure()
            logging.getLogger("uvicorn.error").warning(
                "Fallback event dependency=grading_log_database error=%s "
                "outcome=skip_database_log circuit_opened=%s",
                exc,
                opened,
            )
        else:
            self._breaker.record_success()


def _audit_logger() -> logging.Logger:
    audit_logger = logging.getLogger(_LOGGER_NAME)
    if not audit_logger.handlers:
        handlers: list[logging.Handler] = []
        try:
            database_path = Path(os.getenv("GRADING_LOG_DB_PATH", "logs/grading.db"))
            handlers.append(_SQLiteAuditHandler(database_path))
        except Exception as exc:
            logging.getLogger("uvicorn.error").warning(
                "Fallback event dependency=grading_log_database error=%s "
                "outcome=continue_without_database",
                exc,
            )
        try:
            _LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                _JSON_LOG_PATH,
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(_JsonFormatter())
            handlers.append(file_handler)
        except Exception as exc:
            logging.getLogger("uvicorn.error").warning(
                "Fallback event dependency=grading_json_log error=%s "
                "outcome=continue_without_file",
                exc,
            )
        for handler in handlers or [logging.NullHandler()]:
            audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.DEBUG)
        audit_logger.propagate = False
    return audit_logger


def get_current_request_id() -> str | None:
    """Return the grading request ID while running inside its decorated call."""
    return _REQUEST_ID.get()


def log_grading(function: Callable[P, R]) -> Callable[P, R]:
    """Audit a Grader.grade-style method without changing its return value."""
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        audit_logger = _audit_logger()
        request_data = kwargs.get("request")
        if request_data is None and len(args) > 1:
            request_data = args[1]

        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        student_id = getattr(request_data, "student_id", None)
        common = {
            "request_id": request_id,
            "timestamp": _timestamp(),
            "student_id_hash": _student_hash(student_id),
            "question_id": getattr(request_data, "question_id", None),
            "student_answer": getattr(request_data, "student_answer", None),
            "model_answer": getattr(request_data, "model_answer", None),
            "model_name": getattr(
                getattr(args[0], "_embedding_service", None),
                "model_name",
                os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL),
            ),
            "tokens_used": None,
            "estimated_cost": None,
        }
        context_token: Token[str | None] = _REQUEST_ID.set(request_id)
        try:
            audit_logger.debug(
                "Grading request started",
                extra={
                    "audit_data": {
                        "event_type": "grading_started",
                        "request_id": request_id,
                        "timestamp": common["timestamp"],
                    }
                },
            )
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                entry = {
                    **common,
                    "event_type": "grading",
                    "timestamp": _timestamp(),
                    "ai_score": None,
                    "ai_feedback": None,
                    "response_time_seconds": round(time.perf_counter() - started, 6),
                    "status": "failed",
                    "error_message": str(exc),
                    "stack_trace": traceback.format_exc(),
                }
                audit_logger.error(
                    "Grading request failed",
                    extra={"audit_data": entry},
                )
                raise
            else:
                entry = {
                    **common,
                    "event_type": "grading",
                    "timestamp": _timestamp(),
                    "ai_score": getattr(result, "score", None),
                    "ai_feedback": getattr(result, "feedback_ar", None),
                    "tokens_used": getattr(result, "tokens_used", None),
                    "estimated_cost": getattr(result, "estimated_cost", None),
                    "response_time_seconds": round(time.perf_counter() - started, 6),
                    "status": "success",
                    "error_message": None,
                    "stack_trace": None,
                }
                audit_logger.info(
                    "Grading request completed",
                    extra={"audit_data": entry},
                )
                return result
        finally:
            _REQUEST_ID.reset(context_token)

    return wrapper


def log_teacher_override(
    *,
    request_id: str,
    student_id: str | int | None,
    question_id: str | None,
    old_score: float,
    new_score: float,
    comment: str | None,
) -> None:
    """Record a teacher's score change from an existing override workflow."""
    entry = {
        "event_type": "teacher_override",
        "override_id": str(uuid.uuid4()),
        "timestamp": _timestamp(),
        "request_id": request_id,
        "student_id_hash": _student_hash(student_id),
        "question_id": question_id,
        "old_score": old_score,
        "new_score": new_score,
        "comment": comment,
    }
    _audit_logger().warning(
        "Teacher overrode grading score",
        extra={"audit_data": entry},
    )
