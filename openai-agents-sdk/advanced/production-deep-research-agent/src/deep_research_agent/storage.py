from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import ApprovalDecision, ResearchPlan, ResearchReport, ResearchRequest


class ResearchStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT,
                    findings_json TEXT,
                    sources_json TEXT,
                    evidence_json TEXT,
                    synthesis_json TEXT,
                    report_json TEXT,
                    critique_json TEXT,
                    metrics_json TEXT,
                    approval_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_research_runs_status
                ON research_runs(status);
                """
            )
            existing = {
                row[1] for row in connection.execute("PRAGMA table_info(research_runs)").fetchall()
            }
            for column in ("findings_json", "sources_json", "evidence_json", "synthesis_json"):
                if column not in existing:
                    connection.execute(f"ALTER TABLE research_runs ADD COLUMN {column} TEXT")

    def create_run(self, request: ResearchRequest) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO research_runs(run_id, status, request_json) VALUES (?, ?, ?)",
                (request.run_id, "created", request.model_dump_json()),
            )

    def update(self, run_id: str, status: str, **fields: Any) -> None:
        allowed = {
            "plan_json",
            "findings_json",
            "sources_json",
            "evidence_json",
            "synthesis_json",
            "report_json",
            "critique_json",
            "metrics_json",
            "approval_json",
            "error",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Unsupported storage fields: {sorted(invalid)}")
        assignments = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: list[Any] = [status]
        for key, value in fields.items():
            assignments.append(f"{key} = ?")
            if hasattr(value, "model_dump_json"):
                value = value.model_dump_json()
            elif not isinstance(value, str | None):
                value = json.dumps(value)
            values.append(value)
        values.append(run_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE research_runs SET {', '.join(assignments)} WHERE run_id = ?", values
            )

    def load(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def load_report(self, run_id: str) -> ResearchReport | None:
        row = self.load(run_id)
        if not row or not row.get("report_json"):
            return None
        return ResearchReport.model_validate_json(row["report_json"])

    def load_plan(self, run_id: str) -> ResearchPlan | None:
        row = self.load(run_id)
        if not row or not row.get("plan_json"):
            return None
        return ResearchPlan.model_validate_json(row["plan_json"])

    def load_request(self, run_id: str) -> ResearchRequest | None:
        row = self.load(run_id)
        if not row:
            return None
        return ResearchRequest.model_validate_json(row["request_json"])

    def save_approval(self, run_id: str, decision: ApprovalDecision) -> None:
        self.update(run_id, "approved" if decision.approved else "rejected", approval_json=decision)
