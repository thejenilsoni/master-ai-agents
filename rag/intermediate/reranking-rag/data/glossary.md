# Kestrel Engineering Handbook - Glossary and Error Codes

## Platform terms
- **Beacon** - the internal developer platform: deployments, dashboards, and
  approvals live here.
- **Relay** - the change-capture pipeline that copies application data into the
  warehouse.
- **Waypoint** - the service catalog. Every service has a Waypoint entry naming
  an owning team and a pager target.

## Error codes
- **BCN-503** - Beacon gateway saturation. The deployment API is refusing
  requests because too many rollouts are in flight. Wait and retry; escalate
  when it persists beyond ten minutes.
- **RLY-104** - Relay queue backlog. The warehouse is behind. Analytical data is
  stale but application traffic is unaffected.
- **WPT-401** - Waypoint refused a service registration because the owning team
  field was empty.
