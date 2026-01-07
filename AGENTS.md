# AGENTS.md

## Purpose

This document defines the mandatory development protocol for all coding agents
working on this repository.

It replaces ad-hoc "vibe coding" with a strict, step-gated, spec-driven workflow.
Agents MUST follow this document exactly.

---

## Development Lifecycle (Mandatory)

Every task MUST follow exactly this sequence:

plan -> implement -> test -> review -> doc -> done

No steps may be skipped or reordered.

If test or review fails, the workflow MUST return to implement.

---

## Status Lock (Hard Gate)

The file develop.md MUST contain a status lock on the first line:

Status: <value>

Allowed values (exact spelling):

ready | plan | implement | test | review | doc | done

Rules:

- The user is the only entity allowed to change the Status value
- The agent MUST execute only the step matching the current Status
- If the Status does not match the requested action, the agent MUST stop

This is a hard gate, not a suggestion.

---

## develop.md (Ephemeral Working Log)

develop.md is a single-session, ephemeral development log.

Rules:

- develop.md is NOT tracked by git
- It applies to exactly one feature or task
- At the start of a new task, the agent MUST:
  - Clear all existing content in develop.md
  - Initialize it with the Status lock

Each lifecycle step MUST append its output to develop.md.

---

## Step Responsibilities

### plan

When Status: plan

The agent MUST append a planning section to develop.md containing:

- Confirmation that the entire repository has been read
- Restatement of the user’s requirement in the agent’s own words
- Design decisions and rationale
- How the implementation will:
  - Match existing code style
  - Minimize duplication
  - Use small, reusable functions
- Test design:
  - What will be tested
  - Where tests will live
  - Edge cases considered

No code may be written in this step.

---

### implement

When Status: implement

The agent MUST:

- Implement the planned feature
- Modify production code only (no test code in this step)

The agent MUST append to develop.md:

- Exact files modified
- Which sections or functions were changed
- Why each change was necessary

This section should read like a precise change log, not prose.

---

### test

When Status: test

The agent MUST:

- Implement all required tests for the feature
- Place tests in the appropriate tests/ locations
- Ensure tests are deterministic and isolated

The agent MUST append to develop.md:

- Test files added or modified
- What each test validates
- Important assumptions or fixtures used

IMPORTANT:

- The agent MUST NOT execute tests
- Test execution and result validation are performed by the user
- The agent MUST wait for the user to advance Status

---

### review

When Status: review

The agent MUST perform a conceptual self code review and append results to
develop.md, covering:

- Algorithmic and computational efficiency
- Redundancy and duplication
- Readability and maintainability
- Adherence to project conventions and architecture

IMPORTANT:

- The agent MUST NOT run or assume execution of formatting or linting tools
- Tools such as ruff check and ruff format are run by the user
- The agent reviews the code as-written after user-applied fixes

If review fails, the agent MUST wait for Status to return to implement.

---

### doc

When Status: doc

The agent MUST update README.md to document the new feature.

Rules:

- User-facing features MUST be documented
- Pure refactors may skip documentation unless behavior changes
- Documentation should explain:
  - What the feature does
  - How to use it
  - Where it fits in the pipeline

---

### done

When Status: done

The agent MUST:

- Make no further code changes
- Wait for new instructions

---

## Global Engineering Preferences

These preferences apply globally and always:

- Prefer small, composable, reusable functions
- Avoid hidden side effects
- Avoid duplicated logic
- Favor clarity over cleverness

---

## Repository Structure (Source of Truth)

The agent MUST respect the following directory semantics.

Configuration and secrets:

- ml4investment/config/
  Global configuration and constants
- ml4investment/.env
- ml4investment/.env.example
- ml4investment/tokens.json
  Schwab trading API credentials (DO NOT modify unless explicitly instructed)

Core pipelines:

- ml4investment/backtest.py
  Top-level backtesting entry point
- ml4investment/fetch_data/
  Data fetching logic
- ml4investment/train.py
  Model training entry point
- ml4investment/predict.py
  Prediction and trade execution entry point

Logs and results:

- ml4investment/daily_usage_logs/
  Production environment logs
- ml4investment/logs/
  Development environment logs
- ml4investment/results/
  Detailed experimental outputs (results and logs)
- ml4investment/wandb/
  Weights & Biases artifacts

Experimental artifacts:

- data/
  Experimental datasets, hyperparameters, trained models, analysis plots

Utilities:

- ml4investment/utils/
  Shared helper functions only (no business logic)

---

## Enforcement Principle

If there is any ambiguity:

1. AGENTS.md overrides model defaults
2. User instructions override everything
3. The agent MUST stop and ask if unsure

Failure to follow this protocol is considered a critical error.
