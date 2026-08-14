# CUGA Configuration Directory Recovery

## Goal

Allow `uv run dataset/batch_run_cuga.py` to start CUGA workers and produce
trajectories when `.env` contains blank optional configuration values.

## Root Cause

CUGA uses `CUGA_CONFIGURATIONS_DIR` whenever the environment variable exists.
The current `.env` exports it as an empty string, causing CUGA to resolve
`models/settings.openai.toml` relative to the repository instead of its
installed configurations directory. The CUGA import then fails before the
agent can execute or create a trace.

## Design

- Remove the blank `CUGA_CONFIGURATIONS_DIR` entry from `.env`.
- Before importing CUGA, `run2.py` removes a blank or whitespace-only
  `CUGA_CONFIGURATIONS_DIR` value so inherited shell state cannot recreate the
  failure.
- Preserve non-blank values, allowing callers to intentionally supply a custom
  complete CUGA configuration directory.
- Add a focused regression test for the environment normalization helper.

## Verification

- Run the regression test before and after the implementation.
- Run `uv run dataset/batch_run_cuga.py`.
- Confirm all task results have a non-null `run_id` and copied
  `cuga_trace.json` files.
