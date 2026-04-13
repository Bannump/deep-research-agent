# Aurora Compliance Framework — Overview

The Aurora Compliance Framework (ACF) is a reference model for governing sensitive data pipelines in regulated industries. ACF emphasizes **least-privilege access**, **immutable audit trails**, and **human-in-the-loop approvals** for policy exceptions.

Core principles:
- Data minimization: collect only fields required for the stated purpose.
- Segregation of duties: no single role may both approve and deploy policy changes.
- Continuous verification: automated checks run on every release artifact.

Organizations typically adopt ACF in phases: discovery, control mapping, enforcement hooks, and ongoing attestation.
