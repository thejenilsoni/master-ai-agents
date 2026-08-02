"""
Persistent Chat Sessions (Memory - Beginner)

A conversation buffer that lives in a Python list dies when the process does.
Close the terminal, redeploy the service, or get load-balanced onto a different
worker, and the agent has amnesia. This project moves the transcript out of
process memory and into **SQLite**, keyed by a session id, so a user can quit
mid-conversation and pick it up tomorrow exactly where they left off.

The pattern is the same one every production chat backend uses:

- a `sessions` table (one row per conversation thread),
- a `messages` table with a monotonic `seq` per session (ordering must be stored,
  not inferred from insertion luck),
- an append-only write path: turns are appended, never rewritten,
- reads that rebuild the model's context from durable rows.

`SessionStore` is a plain class over `sqlite3` with no third-party imports, so
every claim about durability is testable offline: write, close the connection,
reopen the file, and prove the same messages come back in the same order.

Run:
    python persistent_sessions.py --selftest              # offline, no key needed
    python persistent_sessions.py --demo                  # offline, no key needed

    export OPENAI_API_KEY="sk-..."
    python persistent_sessions.py --session kitchen-remodel
    python persistent_sessions.py --list
    python persistent_sessions.py --show kitchen-remodel
    python persistent_sessions.py --delete kitchen-remodel
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MODEL = "gpt-4o-mini"

# A generated artifact, not repository content: the file is created on first run
# under a dot-directory so it stays out of the way (and out of commits).
DEFAULT_DB_PATH = ".data/memory.db"

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer in at most four sentences."

# Bounds: the interactive loop, and how much history is replayed to the model.
MAX_CHAT_TURNS = 50
DEFAULT_HISTORY_LIMIT = 20


def _utc_now() -> str:
    """A sortable, timezone-explicit timestamp. Never store naive local time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 1. Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    seq: int = 0
    created_at: str = ""

    def as_dict(self) -> dict[str, str]:
        """The wire format the chat completions API expects."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


# --------------------------------------------------------------------------- #
# 2. The store
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages (session_id, seq);
"""

_VALID_ROLES = frozenset({"system", "user", "assistant"})


class SessionStore:
    """Durable, append-only conversation storage keyed by session id.

    Pass `":memory:"` for a throwaway database (tests, demos) or a filesystem
    path for one that outlives the process.
    """

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = path
        if path != ":memory:":
            # Create the parent directory at runtime so the repo never has to
            # ship a database file.
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # SQLite enforces foreign keys only when asked; without this, deleting a
        # session would silently orphan its messages instead of cascading.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------- #
    def ensure_session(self, session_id: str, title: str = "") -> None:
        """Create the session if it does not exist. Idempotent: resuming an old
        session must never reset its metadata or wipe its messages."""
        if not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        now = _utc_now()
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, title or session_id, now, now),
        )
        self.conn.commit()

    def append(self, session_id: str, role: str, content: str) -> int:
        """Append one message and return its `seq`.

        `seq` is computed and written inside the same transaction as the insert,
        so ordering survives crashes and concurrent writers. Never rely on the
        autoincrement id for per-session ordering - it is global, not per session.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"unknown role: {role!r}")
        self.ensure_session(session_id)
        now = _utc_now()
        with self.conn:  # one transaction: seq lookup + insert + touch
            row = self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row["next_seq"])
            self.conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, seq, role, content, now),
            )
            self.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )
        return seq

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and (via ON DELETE CASCADE) all of its messages."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    # -- reads -------------------------------------------------------------- #
    def history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Return a session's messages in order. `limit` keeps only the newest N."""
        rows = self.conn.execute(
            "SELECT seq, role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        messages = [
            Message(role=r["role"], content=r["content"], seq=r["seq"], created_at=r["created_at"])
            for r in rows
        ]
        if limit is not None and limit >= 0:
            messages = messages[-limit:] if limit else []
        return messages

    def message_count(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])

    def list_sessions(self, limit: int = 100) -> list[SessionInfo]:
        """Most recently updated sessions first - the order a 'resume' menu wants."""
        rows = self.conn.execute(
            """
            SELECT s.session_id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC, s.session_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            SessionInfo(
                session_id=r["session_id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=int(r["message_count"]),
            )
            for r in rows
        ]

    # -- the read the agent actually makes ---------------------------------- #
    def build_context(
        self,
        session_id: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[Message]:
        """Rebuild the model's context from durable rows.

        The system prompt is supplied by the code, not read from the database, so
        you can change the agent's instructions without rewriting stored history.
        Only the newest `limit` messages are replayed - persistence solves
        durability, not context limits.
        """
        return [Message(role="system", content=system_prompt)] + self.history(
            session_id, limit=limit
        )


# --------------------------------------------------------------------------- #
# 3. Offline demo
# --------------------------------------------------------------------------- #
def run_demo(db_path: str) -> None:
    """Prove durability with two independent connections to the same file."""
    session_id = "demo-session"
    print(f"Using database: {db_path}\n")

    # --- "process 1" ------------------------------------------------------- #
    store = SessionStore(db_path)
    store.ensure_session(session_id, title="Planning a balcony garden")
    store.delete_session(session_id)  # start the demo from a clean slate
    store.ensure_session(session_id, title="Planning a balcony garden")
    store.append(session_id, "user", "I want to start a small balcony garden.")
    store.append(session_id, "assistant", "Start with herbs - basil, mint, and thyme survive beginners.")
    store.append(session_id, "user", "My balcony only gets morning sun.")
    store.append(session_id, "assistant", "Then favour mint, parsley, and chives over sun-hungry basil.")
    print("process 1 wrote 4 messages, then exits.")
    store.close()
    print("connection closed - nothing is left in process memory.\n")

    # --- "process 2", started later, maybe on another machine -------------- #
    resumed = SessionStore(db_path)
    print("process 2 opens the same file and resumes the session:")
    for message in resumed.history(session_id):
        print(f"  [{message.seq:>2}] {message.role:<9} {message.content}")

    print("\ncontext that would be sent to the model on the next turn:")
    for message in resumed.build_context(session_id, limit=4):
        print(f"  {message.role:<9} {message.content}")

    print("\nall sessions in the database:")
    for info in resumed.list_sessions():
        print(f"  {info.session_id:<16} {info.message_count:>3} message(s)  updated {info.updated_at}")
    resumed.close()

    print(
        "\nThe conversation outlived the process. That is the whole point:\n"
        "state that matters belongs in a store, not in a variable."
    )


# --------------------------------------------------------------------------- #
# 4. Live chat (the only part that needs an API key)
# --------------------------------------------------------------------------- #
def run_chat(session_id: str, db_path: str, limit: int) -> None:
    """Resume (or start) a session and chat, persisting every turn as it happens."""
    # Deferred imports: --demo and --selftest must work with the standard library.
    import os

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "Install dependencies first: pip install -r requirements.txt\n"
            "(--demo and --selftest need no dependencies at all.)"
        )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or run --demo / --selftest.")

    client = OpenAI()
    store = SessionStore(db_path)
    store.ensure_session(session_id)

    previous = store.history(session_id)
    if previous:
        print(f"Resuming session '{session_id}' with {len(previous)} stored message(s):\n")
        for message in previous[-6:]:
            speaker = "You" if message.role == "user" else "Agent"
            print(f"  {speaker}: {message.content}")
        print()
    else:
        print(f"Starting new session '{session_id}'.\n")

    print("Type 'exit' to quit. Everything you say is saved as it happens.\n")
    try:
        for _ in range(MAX_CHAT_TURNS):
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            # Persist the user turn *before* calling the model: if the API call
            # fails, the user's message is not lost.
            store.append(session_id, "user", user_input)

            context = store.build_context(session_id, limit=limit)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[m.as_dict() for m in context],
            )
            reply = (response.choices[0].message.content or "").strip()
            store.append(session_id, "assistant", reply)
            print(f"Agent: {reply}\n")
    finally:
        store.close()
        print(f"Session '{session_id}' saved to {db_path}. Rerun with the same "
              f"--session to continue it.")


# --------------------------------------------------------------------------- #
# 5. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Prove durability, ordering, isolation, and cascade deletes - offline."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "nested" / "memory.db")

        # -- process 1: write, then close the connection -------------------- #
        store = SessionStore(db_path)
        assert Path(db_path).exists(), "the store must create its parent directory"
        store.ensure_session("alpha", title="First conversation")
        assert store.append("alpha", "user", "Remember that I prefer trains over flights.") == 1
        assert store.append("alpha", "assistant", "Noted - trains it is.") == 2
        assert store.append("alpha", "user", "Plan me a two-city trip.") == 3
        store.append("beta", "user", "A completely separate conversation.")
        store.close()

        # -- process 2: reopen the same file -------------------------------- #
        resumed = SessionStore(db_path)
        history = resumed.history("alpha")
        assert len(history) == 3, history
        assert [m.seq for m in history] == [1, 2, 3], "seq must be contiguous and ordered"
        assert [m.role for m in history] == ["user", "assistant", "user"]
        assert history[0].content == "Remember that I prefer trains over flights."
        assert all(m.created_at for m in history), "every row must be timestamped"

        # -- sessions are isolated ------------------------------------------ #
        assert len(resumed.history("beta")) == 1
        assert resumed.history("does-not-exist") == []

        # -- ensure_session is idempotent ----------------------------------- #
        resumed.ensure_session("alpha", title="A different title")
        assert len(resumed.history("alpha")) == 3, "resuming must not wipe history"
        titles = {s.session_id: s.title for s in resumed.list_sessions()}
        assert titles["alpha"] == "First conversation", "metadata must not be reset"

        # -- listing --------------------------------------------------------- #
        listed = {s.session_id: s.message_count for s in resumed.list_sessions()}
        assert listed == {"alpha": 3, "beta": 1}, listed

        # -- context rebuild: system prompt first, newest history after ------ #
        context = resumed.build_context("alpha", system_prompt="SYSTEM", limit=2)
        assert context[0].role == "system" and context[0].content == "SYSTEM"
        assert [m.content for m in context[1:]] == [
            "Noted - trains it is.",
            "Plan me a two-city trip.",
        ]
        assert resumed.build_context("alpha", limit=0)[1:] == []

        # -- appends keep numbering after a resume --------------------------- #
        assert resumed.append("alpha", "assistant", "Here is a draft itinerary.") == 4

        # -- validation ------------------------------------------------------ #
        for bad in (lambda: resumed.append("alpha", "wizard", "nope"),
                    lambda: resumed.ensure_session("   ")):
            try:
                bad()
                raise AssertionError("invalid input should have been rejected")
            except ValueError:
                pass

        # -- delete cascades -------------------------------------------------- #
        assert resumed.delete_session("alpha") is True
        assert resumed.history("alpha") == []
        assert resumed.message_count("alpha") == 0, "messages must cascade with the session"
        assert resumed.delete_session("alpha") is False, "deleting twice is not an error"
        assert len(resumed.history("beta")) == 1, "deleting one session must not touch another"
        resumed.close()

        # -- process 3: the delete is durable too ---------------------------- #
        final = SessionStore(db_path)
        assert final.history("alpha") == []
        assert len(final.history("beta")) == 1
        final.close()

    print("selftest passed:")
    print("  - a session written by one connection resumes intact in another")
    print("  - per-session seq numbering is contiguous, ordered, and survives resume")
    print("  - sessions are isolated; ensure_session is idempotent")
    print("  - deleting a session cascades to its messages and is durable")


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat sessions persisted to SQLite and resumable by session id."
    )
    parser.add_argument("--selftest", action="store_true", help="verify the store offline")
    parser.add_argument("--demo", action="store_true", help="offline durability walkthrough")
    parser.add_argument("--session", default="default", help="session id to start or resume")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_HISTORY_LIMIT, help="messages replayed to the model"
    )
    parser.add_argument("--list", action="store_true", help="list stored sessions and exit")
    parser.add_argument("--show", metavar="SESSION_ID", help="print a stored transcript and exit")
    parser.add_argument("--delete", metavar="SESSION_ID", help="delete a session and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.demo:
        run_demo(args.db)
        return

    if args.list or args.show or args.delete:
        with SessionStore(args.db) as store:
            if args.list:
                sessions = store.list_sessions()
                if not sessions:
                    print(f"No sessions in {args.db} yet.")
                for info in sessions:
                    print(
                        f"{info.session_id:<24} {info.message_count:>4} message(s)  "
                        f"updated {info.updated_at}  {info.title}"
                    )
            if args.show:
                messages = store.history(args.show)
                if not messages:
                    print(f"No stored messages for session '{args.show}'.")
                for message in messages:
                    print(f"[{message.seq:>3}] {message.role:<9} {message.content}")
            if args.delete:
                deleted = store.delete_session(args.delete)
                print(f"{'Deleted' if deleted else 'No such session:'} {args.delete}")
        return

    run_chat(args.session, args.db, args.limit)


if __name__ == "__main__":
    main()
