# Kestrel Engineering Handbook - Deployments

## Release trains
Services ship on a daily release train. The train builds at 09:00 local time,
runs the full test suite, and promotes the artifact to staging automatically.
Promotion from staging to production is a manual approval in Beacon, and the
approver must be someone other than the change author.

## Canary releases
Production rollouts are canaried. Beacon sends five percent of traffic to the
new version for fifteen minutes, then twenty-five percent, then everything.
Error rate and latency are compared against the previous version at each step,
and the rollout halts automatically if the error rate doubles.

## Rolling back
To undo a bad release, run `beacon rollback <service> --to previous`. A rollback
restores the last known good artifact together with the configuration it shipped
with, because configuration drift is the most common cause of a failed recovery.
Rollbacks never require an approval; rolling back quickly is always preferred to
debugging in production.

## Deployment freezes
A freeze blocks promotion to production. Freezes are declared for company-wide
events and for the last two business days of every quarter. During a freeze only
changes that fix an active incident may ship, and each one needs an incident
commander's sign-off.
