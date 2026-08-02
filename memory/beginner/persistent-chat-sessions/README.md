# Persistent Chat Sessions (Durable Session State in SQLite)

A **beginner** project on the difference between an agent that remembers and an
agent that merely *has not crashed yet*. A conversation buffer living in a Python
list dies with the process: close the terminal, redeploy the service, or get
load-balanced onto a different worker, and your agent has amnesia.

Here the transcript moves out of process memory and into **SQLite**, keyed by a
session id. Quit mid-conversation, come back tomorrow, pass the same
`--session`, and the agent picks up exactly where you left off.

This advances on
[Conversation Buffer Memory](../conversation-buffer-memory): the same message
list, now stored durably outside the process instead of vanishing when it exits.

## What it demonstrates

- **Session-keyed state** — one `session_id` (or thread id) per conversation, the
  same idea every production chat backend and checkpointer is built on.
- **Explicit ordering** — a monotonic `seq` per session written inside the insert
  transaction. Ordering is stored, never inferred from insertion luck, and never
  taken from the global autoincrement id.
- **Append-only writes** — turns are appended and never rewritten, which keeps
  the transcript auditable and makes concurrent writers safe.
- **Rebuilding context from rows** — `build_context()` reassembles what the model
  sees from durable storage, with the system prompt supplied by *code* so you can
  change the agent's instructions without rewriting stored history.
- **Persist before you call** — the user's message is committed before the API
  call, so a failed request never loses what they typed.
- **Cascade deletes** — `PRAGMA foreign_keys = ON` plus `ON DELETE CASCADE`, so
  deleting a session actually deletes its messages instead of orphaning them.

## The schema

| Table | Columns |
| --- | --- |
| `sessions` | session_id (PK), title, created_at, updated_at |
| `messages` | id (PK), session_id (FK, cascade), seq, role, content, created_at |

`UNIQUE (session_id, seq)` is what guarantees the transcript can always be
replayed in the order it happened.

The database is created at runtime at `.data/memory.db` (override with `--db`).
It is a generated artifact — nothing is committed with the project.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/memory/beginner/persistent-chat-sessions
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# Start (or resume) a named conversation:
python persistent_sessions.py --session kitchen-remodel

# Quit, then run the exact same command tomorrow to continue it.

python persistent_sessions.py --list                    # every stored session
python persistent_sessions.py --show kitchen-remodel    # the full transcript
python persistent_sessions.py --delete kitchen-remodel  # forget it entirely
```

## Verify it without an API key

The store is plain `sqlite3` with a built-in self-test that writes to a temporary
database, closes the connection, reopens the file, and checks the conversation
came back intact:

```bash
python persistent_sessions.py --selftest
# selftest passed:
#   - a session written by one connection resumes intact in another
#   - per-session seq numbering is contiguous, ordered, and survives resume
#   - sessions are isolated; ensure_session is idempotent
#   - deleting a session cascades to its messages and is durable
```

There is also an offline walkthrough that simulates two separate processes
sharing one database file:

```bash
python persistent_sessions.py --demo
```

## Example session

```
$ python persistent_sessions.py --session balcony-garden
Starting new session 'balcony-garden'.

You: I want to start a small balcony garden.
Agent: Start with herbs — basil, mint, and thyme survive beginners.

You: exit
Session 'balcony-garden' saved to .data/memory.db.

# ... the next day, a brand-new process ...

$ python persistent_sessions.py --session balcony-garden
Resuming session 'balcony-garden' with 2 stored message(s):

  You: I want to start a small balcony garden.
  Agent: Start with herbs — basil, mint, and thyme survive beginners.

You: My balcony only gets morning sun.
Agent: Then favour mint, parsley, and chives over sun-hungry basil.
```

Note that `--limit` still caps how much of that history is replayed to the model.
Persistence solves durability; it does **not** solve context limits — the
transcript now grows forever *and* survives restarts, which makes the trimming
from the previous project more necessary, not less.

## Extending this project

- Add a `user_id` column and scope sessions to it, so one database serves many
  users.
- Store token counts per message at write time and combine this with
  `trim_to_token_budget` from
  [Conversation Buffer Memory](../conversation-buffer-memory).
- Swap SQLite for Postgres — the interface (`append`, `history`, `build_context`)
  does not change, which is why it is worth having an interface.
- Add a `deleted_at` column for soft deletes when you need an audit trail.
- Compress old turns instead of replaying only the newest ones — that is
  [Summarizing Memory](../../intermediate/summarizing-memory).
