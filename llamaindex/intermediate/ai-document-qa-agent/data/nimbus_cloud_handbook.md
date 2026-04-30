# Nimbus Cloud — Product & Support Handbook

## About Nimbus Cloud

Nimbus Cloud is a developer-first cloud platform offering managed compute,
object storage, and a serverless functions runtime. It was founded in 2021 and
serves more than 8,000 teams worldwide.

## Products

### Nimbus Compute
On-demand virtual machines billed per second. Instance families:
- **n-micro** — 1 vCPU, 1 GB RAM. Good for side projects.
- **n-standard** — 2 vCPU, 8 GB RAM. The default for web services.
- **n-compute** — 8 vCPU, 16 GB RAM. CPU-optimized for batch jobs.

### Nimbus Store
S3-compatible object storage with 11 nines of durability. The first 10 GB of
storage and 50 GB of egress per month are free on every plan.

### Nimbus Functions
A serverless runtime supporting Python, Node.js, and Go. Functions scale to zero
when idle and cold starts are typically under 250 ms.

## Pricing Plans

| Plan | Monthly price | Included compute hours | Support |
| --- | --- | --- | --- |
| Free | $0 | 50 | Community forum |
| Pro | $49 | 500 | Email, 24h response |
| Scale | $199 | 2,500 | Priority email + chat, 4h response |
| Enterprise | Custom | Custom | Dedicated TAM, 1h response |

Overage compute beyond the included hours is billed at $0.04 per vCPU-hour.

## Support & SLA

- The **Scale** and **Enterprise** plans include a 99.95% uptime SLA. If monthly
  uptime drops below this, customers receive service credits: 10% for uptime
  between 99.0% and 99.95%, and 25% for uptime below 99.0%.
- Support hours for Pro are 9am–6pm UTC on business days. Scale and Enterprise
  receive 24/7 coverage.
- Security issues can be reported to security@nimbus.example and are
  acknowledged within 2 hours, 24/7.

## Refunds

Annual plans can be cancelled within 30 days for a full refund. Monthly plans
are non-refundable but can be cancelled at any time to stop future billing.

## Data Residency

Customers may choose to store data in the US, EU (Frankfurt), or AP (Singapore)
regions. Enterprise customers can request additional regions.
