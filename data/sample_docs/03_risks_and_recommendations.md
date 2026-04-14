# Risks and Recommendations

## Common failure modes
- **Shadow pipelines**: undeployed scripts that bypass official ingestion paths.
- **Stale policies**: controls documented but not enforced in CI/CD.
- **Alert fatigue**: excessive notifications causing operators to ignore true positives.

## Recommendations
1. Treat policy-as-code: version policies alongside application code.
2. Run periodic tabletop exercises for breach and rollback scenarios.
3. Measure control effectiveness with sampled audits and automated conformance tests.

## Future direction
The working group is evaluating confidential computing for highly sensitive analytics where data must remain encrypted during processing.
