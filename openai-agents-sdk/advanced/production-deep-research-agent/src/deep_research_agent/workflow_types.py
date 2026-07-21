from __future__ import annotations

from typing import Awaitable, Callable

from .citations import CitationAudit
from .models import ApprovalDecision, Critique, ResearchReport

ApprovalCallback = Callable[[ResearchReport, Critique, CitationAudit], Awaitable[ApprovalDecision]]
ProgressCallback = Callable[[str, str], None]
