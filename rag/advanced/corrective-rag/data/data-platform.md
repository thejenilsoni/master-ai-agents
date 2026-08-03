# Kestrel Engineering Handbook - Data Platform

## Where data lives
Application databases are the source of truth. Analytical copies land in the
warehouse through Relay, the change-capture pipeline, usually within ten
minutes of the original write.

## Retention
Raw event data is kept for ninety days. Aggregated daily tables are kept for
three years. Anything containing customer identifiers must be written to a table
tagged `pii`, and tagged tables are dropped automatically once their retention
window expires.

## Access
Warehouse access is granted per dataset through the access portal and is
reviewed every quarter. Access to `pii` tables additionally requires manager
approval and lapses after thirty days unless it is renewed.

## Schema changes
Adding a column is safe and needs no notice. Renaming or dropping a column
requires a deprecation notice in the data channel two weeks ahead, because
dashboards and downstream jobs bind to column names.
