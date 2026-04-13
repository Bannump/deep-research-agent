# Operating Level Agreement — Research & Analytics Platform

**Effective:** FY2026  
**Parties:** Business stakeholders and Platform Engineering

## Availability targets

- **Production API:** 99.5% monthly uptime, excluding scheduled maintenance (max 4 hours/month, announced ≥72 hours in advance).
- **Batch analytics jobs:** Daily SLA window 02:00–06:00 local; missed runs must be replayed within 24 hours.

## Support tiers

| Severity | Definition                         | Response time | Workaround target |
|----------|-------------------------------------|---------------|-------------------|
| P1       | Production down or data loss risk   | 15 minutes    | 2 hours           |
| P2       | Major degradation, workaround exists| 1 hour        | 8 hours           |
| P3       | Minor issue                         | 1 business day| Best effort       |

## Capacity & fairness

- Interactive queries are subject to **per-tenant concurrency caps** to prevent noisy-neighbor effects.
- Long-running research jobs may be **queued** during peak hours; tenants receive position-in-queue estimates when available.

## Change management

Production changes require peer review, automated tests, and a rollback plan. Emergency changes require post-incident review within five business days.
