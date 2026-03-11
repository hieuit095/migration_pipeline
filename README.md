# AI-Powered Legacy Code Migration Pipeline

## Overview
A high-performance, multi-agent production line for automating the migration of legacy codebases to modern stacks. Built with Rust for safety and concurrency, utilizing the ZeroClaw agent OS.

The pipeline now supports multiple inference providers and task-level model routing. Each task can point at a different provider/model pair without code changes.

## Architecture
The pipeline utilizes a non-linear feedback loop with four specialized agents:

1.  **Blueprinter (`google/gemini-3-flash-preview`)**: Analyzes the legacy codebase and generates a migration strategy (Tickets).
2.  **Executor (`Minimax M2.5`)**: Translates legacy code blocks into the modern target stack.
3.  **Verifier (`GLM-5`)**: Validates semantic equivalence, generates tests, and runs isolated Docker-based syntax checks.
4.  **Surgeon (`Claude 3.5 Sonnet`)**: Performs deep recursive debugging for complex failures and feeds fixes back into verification.

## Directory Structure
- `src/main.rs`: Entry point and orchestration.
- `src/config/`: Configuration for LLMs and API keys.
- `src/agents/`: Agent implementations and traits.
- `src/skills/`: Tooling for AST parsing (Tree-sitter) and File I/O.
- `src/pipeline/`: Core feedback loop logic.
- `src/utils/`: Telemetry and logging utilities.
- `modern_app/`: Generated modernized source output written by the executor.
- `logs/`: Daily rotating pipeline and audit logs, including `pipeline.log`.

## Getting Started

### Prerequisites
- Rust 1.83+
- Docker with a running daemon
- OpenRouter API Key and/or Together API Key

### Build
```bash
cargo build
```

### Configuration
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_api_key_here
TOGETHER_API_KEY=your_together_api_key_here

# Per-task provider/model routing
# Supported providers: openrouter, together
BLUEPRINTER_PROVIDER=openrouter
BLUEPRINTER_MODEL=google/gemini-3-flash-preview

# Future tasks can use the same pattern:
# EXECUTOR_PROVIDER=openrouter
# EXECUTOR_MODEL=minimax/minimax-m2.5
# VERIFIER_PROVIDER=openrouter
# VERIFIER_MODEL=z-ai/glm-5
# SURGEON_PROVIDER=openrouter
# SURGEON_MODEL=anthropic/claude-3.5-sonnet
```

If a task-specific provider/model override is omitted, the code falls back to the built-in default for that agent. Today the live flow uses the blueprinter route; the same env naming convention is already in place for executor, verifier, and surgeon as those agents come online.

When you switch a task to a different provider, set both `<TASK>_PROVIDER` and `<TASK>_MODEL` together so the provider and model stay compatible.

Phase 2 now writes executor output under `modern_app/`, creating any missing parent directories automatically before each generated file is saved.

Phase 3 adds a verifier pass that writes generated tests under `modern_app/tests/` before a ticket can be marked `Verified`.

Phase 4 adds a non-linear feedback loop: failed tickets are routed through the surgeon, then bounced back to the verifier, with a hard retry cap of `3` surgeries per ticket.

Phase 5 adds persistent checkpointing to `.migration_state.json`. The pipeline now resumes from the saved ticket state on startup and uses a dedicated Tokio MPSC-backed state writer to atomically checkpoint every ticket transition.

Phase 6 adds observability and auditability. Every LLM call now records provider/model metadata, prompt payloads, exact responses, and token usage in rotating `logs/pipeline.log.*` files, and each `Ticket` persists cumulative token usage in `.migration_state.json` for downstream cost reporting.

Phase 7 removes direct host execution from verification. All syntax checks now run in short-lived Docker containers with `--network=none`, strict CPU and memory limits, and a read-only `modern_app` mount (`:ro`) to reduce code execution risk.
