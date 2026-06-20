## Description of Changes

Please summarize the changes implemented in this Pull Request and detail the problem/feature they resolve.

## Linked Issues / Tickets

Closes # (issue number)

## Production Quality Checklist

Before submitting this PR, please check that you have completed the following steps:

- [ ] **Linting & Formatting**: Verified that Ruff passes cleanly without errors or warnings (`ruff check .` and `ruff format --check .`).
- [ ] **Type Safety**: Checked that MyPy passes cleanly without static typing issues (`mypy .`).
- [ ] **Security Scans**: Scanned codebase using Bandit and verified no high-severity vulnerabilities are found (`bandit -r token_diet/`).
- [ ] **Unit Tests**: Executed the test suite locally and confirmed all tests pass (`pytest`).
- [ ] **Test Coverage**: Inspected test coverage to ensure no untested regressions are introduced (`pytest --cov=token_diet`).

## Testing Documentation

Please describe how you verified these changes (e.g. mock runs, manual scripts, or specific log outputs).
