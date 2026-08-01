# Contributing

Thanks for helping make embodied clinical AI more observable and testable.

## Good first contributions

- Add deterministic hospital scenarios and assertions.
- Improve collision geometry, navigation metrics, or evidence capture.
- Add adapters against public sandbox APIs—never real patient systems.
- Improve installation on Linux or document reproducible hardware results.
- Tighten safety checks, privacy boundaries, and failure reporting.

## Development workflow

1. Fork the repository and create a focused branch.
2. Run `scripts/setup_macos.sh` or install the Python dependencies manually.
3. Make the smallest coherent change and add or update tests.
4. Run:

   ```bash
   .venv/bin/pytest -q
   bash -n scripts/*.sh
   python -m compileall -q baymax_nurse tests
   ```

5. Open a pull request explaining the behavior change, safety impact, and test
   evidence.

## Guardrails

- Do not include patient data, API keys, credentials, or proprietary hospital
  information in code, fixtures, screenshots, issues, or logs.
- Keep every example explicitly simulation-only.
- Do not present generated observations as diagnoses or treatment guidance.
- Do not commit downloaded meshes or policies unless their license permits
  redistribution and the notice is updated.
- New external actions must be authenticated, idempotent, auditable, and easy to
  replace with a local dummy in tests.

For substantial architecture changes, open a feature request first so the
design and safety boundary can be discussed before implementation.
