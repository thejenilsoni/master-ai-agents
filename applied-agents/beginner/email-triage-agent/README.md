# Email Triage Agent

The real cost of a busy inbox is not reading it — it is the two threads you
never got to and the promise you made on Tuesday that you have now forgotten.
This agent reads a mailbox and produces a worklist: every thread sorted into
`urgent` / `needs-reply` / `fyi` / `spam`, a list of **commitments you made in
your own earlier replies**, and a drafted response for anything that needs one,
written in a tone you configure.

**It never sends anything.** There is no SMTP client, no mail API, no `--send`
flag, and no credential for one. The only outputs are text on stdout and an
optional Markdown file you review yourself. The self-test asserts that the
source contains no mail-sending code at all, so the guarantee is checked rather
than promised.

## What it demonstrates

- **Rules before the model** — `prefilter()` classifies obvious phishing and
  no-reply notifications with plain Python, so three of the eight sample threads
  never reach the API. Triage cost scales with the *ambiguous* mail, not the
  total.
- **Phishing signals that actually work** — a display name claiming to be your
  own company from a domain that is not your company's is the strongest tell,
  and it is one line of deterministic code.
- **Configurable tone as data, not prose** — presets live in `tones.json`,
  validated into a `ToneConfig`, override-able per run with `--set key=value`.
  An unknown preset or an unknown field is a hard error, because silently
  replying in the wrong voice for a week is worse than crashing.
- **Commitment extraction with evidence** — every promise must quote the
  sentence *you* wrote. No quote, no commitment.
- **Drafts that admit ignorance** — the model is told never to invent a date or
  write a `[placeholder]`; it must put unknowns in `open_items` instead.
  `find_placeholders()` then catches any that slip through and flags the draft
  **needs human review**.
- **Bounded cost** — at most `MAX_EMAILS` threads read and `MAX_DRAFTS` replies
  written per run.

## The sample mailbox

`sample_mailbox.json` is an entirely invented inbox for Priya Raman at the
fictional Harborline Systems: eight threads including a customer escalation with
a promise she already made, a vendor renewal question, two automated
notifications, a phishing attempt, and a marketing blast. Every address uses the
reserved `.example` domain.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/beginner/email-triage-agent
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
# Default tone (professional), bundled mailbox:
python email_triage_agent.py

# Pick a tone and tweak one field:
python email_triage_agent.py --tone warm --set sign_off="Cheers,"

# Terse replies, saved to a file you can review:
python email_triage_agent.py --tone brief --write-drafts drafts.md

# Your own mailbox export, same JSON shape:
python email_triage_agent.py --mailbox /path/to/mailbox.json
```

Available tones: `professional` (default), `warm`, `brief`, `detailed-formal`.
Any field of a preset can be overridden: `formality`, `length`, `greeting`,
`sign_off`, `use_first_names`, `guidelines`.

## Verify it without an API key

Mailbox validation, the prefilter rules, tone resolution, placeholder detection,
signature handling and worklist ranking are all pure functions:

```bash
python email_triage_agent.py --selftest
# selftest passed:
#   8 threads loaded and validated for Priya Raman
#   prefilter caught 2 spam + 1 automated thread before any API call
#   tone presets: brief, detailed-formal, professional, warm (default: professional)
#   placeholder detection, signature handling and ranking all correct
```

## Example output

```markdown
# Inbox triage for Priya Raman

**Tone:** professional (neutral, standard)
**Threads triaged:** 8
**Nothing was sent.** Every reply below is a draft.

## Worklist

| # | Category | From | Subject | Deadline signals |
| --- | --- | --- | --- | --- |
| 1 | urgent | Nadia Sorensen | Speaker submissions close tomorrow | tomorrow, deadline |
| 2 | urgent | Tomas Brandt | Atlas Console exports failing since the 04:00 job | this morning, today |
| 3 | needs-reply | Elena Vasquez | Renewal paperwork for the Loomstack data feed | — |
| 4 | needs-reply | Mei Okafor | Re: Review request - permissions migration runbook | — |
| 5 | fyi | Harborline CI | Build #4821 failed on main | — |
| 7 | spam | "Harborline Systems IT Security" | URGENT ACTION REQUIRED ON YOUR ACCOUNT | within 24 hours |

## Commitments you made

- **Send Fairhaven a written status update on the export failures** (by 5pm on 2026-03-09)
  > I'll send you a written status update by 5pm tomorrow, and if we can't identify
  > the cause by then I'll set up a call with the engineer who owns the export pipeline.
- **Review the escalation section of the rollback runbook and leave comments** (by Wednesday)
  > I will review the escalation section and leave comments by Wednesday.

## Drafted replies (not sent)

### Atlas Console exports failing since the 04:00 job ⚠️ needs human review

**To:** Tomas Brandt <tomas.brandt@fairhaven-logistics.example>
**Subject:** Re: Atlas Console exports failing since the 04:00 job

    Hi Tomas,

    You're right that I owe you the status update I promised by 5pm yesterday,
    and I'm sorry it's late. The export job has now failed three consecutive
    nights and the platform team has it as their active incident.

    On switching back to the legacy format in the meantime: I don't want to
    answer that from guesswork, so I'm confirming with the engineer who owns
    the pipeline and will come back to you today either way.

    Best,
    Priya Raman

Left for you to answer:
- Whether Fairhaven should temporarily revert to the legacy export format
- The current root-cause status from the platform team
```

Note what the draft did *not* do: it did not invent a root cause, and it did not
answer the legacy-format question. Both went into `open_items`, which is what
flipped the thread to **needs human review**.

## Extending this project

- Point `load_mailbox()` at a read-only export from your own mail client; the
  JSON shape is four fields per message.
- Add your own prefilter rules — internal mailing lists, ticket-system
  notifications, anything you already ignore by reflex.
- Save the drafts to your mail client's *drafts* folder yourself after reading
  them; keep the agent out of the send path.
- Track commitments across runs in SQLite and report the ones that have aged
  past their stated date.
- Add a per-sender tone override so your biggest customer always gets the
  `detailed-formal` voice.
