# Third-Party Security Review Checklist (Internal)

**Document owner:** Enterprise Risk  
**Applies to:** Vendors with access to customer data or production adjacent systems  
**Review cadence:** Annual, or within 90 days of material architecture change

## Minimum evidence package

1. **Architecture diagram** showing data flows, trust boundaries, and integration points.
2. **Access control model** including SSO/SAML configuration, MFA enforcement for admin roles, and break-glass procedures.
3. **Penetration test summary** from the last 12 months, with critical/high findings remediated or risk-accepted with executive sign-off.
4. **Subprocessor list** aligned with the vendor’s published policy, including cross-border transfer mechanisms where applicable.

## Scoring rubric (simplified)

| Area            | Pass criteria                                      |
|-----------------|----------------------------------------------------|
| Identity        | No shared admin accounts; quarterly access reviews |
| Logging         | Centralized logs; tamper resistance for audit logs |
| Incident comms  | Defined SLAs for P1/P2; customer notification path   |
| Data handling   | Encryption in transit and at rest; key custody   |

## Escalation

Findings rated **Critical** or **High** with no remediation plan within 30 days must be escalated to the Chief Information Security Officer and may block procurement renewal.
