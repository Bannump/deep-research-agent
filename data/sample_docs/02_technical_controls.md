# Technical Controls

## Encryption
Data at rest must use AES-256 or equivalent. Data in transit must use TLS 1.2+ with modern cipher suites. Key management should integrate with a hardware security module (HSM) or cloud KMS with quarterly rotation policies.

## Access control
Access decisions combine role-based access control (RBAC) with attribute-based constraints (ABAC) such as project membership and data sensitivity labels. Emergency break-glass access is permitted but must trigger alerts and post-incident review within 48 hours.

## Logging
Centralized logging is mandatory. Logs must include actor identity, action, resource, timestamp, and correlation identifiers. Tamper-evident storage is recommended for audit-grade events.
