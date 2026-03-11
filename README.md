# AI-Powered Legacy Code Migration Pipeline

## Overview
A high-performance, multi-agent production line for automating the migration of legacy codebases to modern stacks. Built with Rust for safety and concurrency, utilizing the ZeroClaw agent OS.

The pipeline now uses `zeroclaw` for OpenRouter-backed model routing and native structured tool calls. Each task can point at a different OpenRouter model without code changes.

## Architecture
The pipeline utilizes a non-linear feedback loop with four specialized agents:

1.  **Blueprinter (`google/gemini-3-flash-preview`)**: Analyzes the legacy codebase and generates a migration strategy (Tickets).
2.  **Executor (`Minimax M2.5`)**: Translates legacy code blocks into the modern target stack and generates matching tests.
3.  **Verifier (`z-ai/glm-5`)**: Generates shadow-test fixtures from the legacy AST, executes both legacy and modern code in isolated containers, and rejects any behavioral divergence.
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
- Rust 1.87+
- Docker with a running daemon
- OpenRouter API Key

### Build
```bash
cargo build
```

### Configuration
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_api_key_here

# Per-task model routing
BLUEPRINTER_MODEL=google/gemini-3-flash-preview

# Optional per-task overrides:
# EXECUTOR_MODEL=minimax/minimax-m2.5
# VERIFIER_MODEL=z-ai/glm-5
# SURGEON_MODEL=anthropic/claude-3.5-sonnet
```

If a task-specific model override is omitted, the code falls back to the built-in default for that agent.

Phase 2 now writes executor output under `modern_app/`, creating any missing parent directories automatically before each generated file is saved.

Phase 3 and later stages now use shadow testing as the primary quality gate. The verifier generates structured JSON fixtures from the legacy AST, runs the legacy and modern targets side-by-side in Docker, and records a strict execution diff before any ticket can be marked `Verified`.

Phase 4 adds a non-linear feedback loop: failed tickets are routed through the surgeon, then bounced back to the verifier, with a hard retry cap of `3` surgeries per ticket.

Phase 5 adds persistent checkpointing to `.migration_state.json`. The pipeline now resumes from the saved ticket state on startup and uses a dedicated Tokio MPSC-backed state writer to atomically checkpoint every ticket transition.

Phase 6 adds observability and auditability. Every LLM call now records provider/model metadata, prompt payloads, structured responses, and token usage in rotating `logs/pipeline.log.*` files, and each `Ticket` persists cumulative token usage in `.migration_state.json` for downstream cost reporting.

Phase 7 removes direct host execution from verification. All syntax checks now run in short-lived Docker containers with `--network=none`, strict CPU and memory limits, and a read-only `modern_app` mount (`:ro`) to reduce code execution risk.

The current LLM architecture no longer uses the custom `reqwest` gateway. Blueprinter, Executor, Verifier, and Surgeon call OpenRouter through `zeroclaw`, consume native tool-call payloads, and rely on gateway-level structured-output retries instead of local string-cleaning helpers.

Current shadow execution runtime support is implemented for JavaScript, TypeScript, and Python entry points. Unsupported target runtimes fail fast with a descriptive error so they can be extended with project-specific harnesses.
