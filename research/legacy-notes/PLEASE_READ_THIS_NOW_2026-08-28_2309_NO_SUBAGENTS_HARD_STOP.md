# OWNER HARD STOP — NO SUBAGENTS / NO DELEGATED AGENTS

This is not advisory.

**NO SUBAGENTS ARE ALLOWED FOR THIS WORK.**

Do not invoke or spawn:
- `codex exec`
- additional Codex workers
- Claude / OpenCode
- sub-agents
- code-mode delegated workers
- agent loops
- autonomous coding/research wrappers
- parallel delegated reasoning/coding jobs

The primary worker must perform inspection, research, edits, builds, benchmarks, profiling, and classification directly.

This rule already exists in `AGENTS.md` and earlier owner notes. The multiple detached `codex exec` workers observed around 23:05–23:08 on 2026-08-28 were a violation of that explicit instruction.

Do not reinterpret “worker”, “helper”, “researcher”, “code-mode”, detached process, or background agent as an exception. There is no exception.

Ordinary host tools/build processes (`cmake`, `ninja`, `nvcc`, benchmark binaries, shell scripts, Python analysis scripts) are fine. **A second reasoning/coding agent is not.**

If a task is too large, do it sequentially in the primary worker. Do not delegate it.
