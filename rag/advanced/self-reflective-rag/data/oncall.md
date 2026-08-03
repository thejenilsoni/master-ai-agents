# Kestrel Engineering Handbook - On-call and Incidents

## The rotation
Each service team runs a weekly on-call rotation with a primary and a secondary.
Handover happens Monday morning and includes a written summary of anything still
open. The secondary is paged only when the primary does not acknowledge within
five minutes.

## Severity levels
- **SEV1** - customer-facing outage or data loss. Page immediately, open an
  incident channel, and assign an incident commander.
- **SEV2** - major degradation with a workaround available. Page during business
  hours only.
- **SEV3** - minor or internal-only impact. File a ticket; nobody is paged.

## What the on-call engineer does first
Acknowledge the page, then open the service dashboard in Beacon and look at the
error budget and the most recent deployment. Most alerts fire shortly after a
change, so the first question is always "what shipped?" Roll back first and
investigate afterwards.

## Incident review
Every SEV1 and SEV2 gets a written review within five business days. Reviews are
blameless and concentrate on the conditions that allowed the failure rather than
on the person who typed the command. Action items get an owner and a due date
and are tracked to completion.
