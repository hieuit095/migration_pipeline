# AI-Powered Legacy Code Migration Pipeline

## Overview
A high-performance, multi-agent production line for automating the migration of legacy codebases to modern stacks. Built with Rust for safety and concurrency, utilizing the ZeroClaw agent OS.

## Architecture
The pipeline utilizes a non-linear feedback loop with four specialized agents:

1.  **Blueprinter (`google/gemini-3-flash-preview`)**: Analyzes the legacy codebase and generates a migration strategy (Tickets).
2.  **Executor (`Minimax M2.5`)**: Translates legacy code blocks into the modern target stack.
3.  **Verifier (`GLM-5`)**: Validates semantic equivalence and runs sandboxed tests.
4.  **Surgeon (`Claude 4.5`)**: Performs deep recursive debugging for complex failures.

## Directory Structure
- `src/main.rs`: Entry point and orchestration.
- `src/config/`: Configuration for LLMs and API keys.
- `src/agents/`: Agent implementations and traits.
- `src/skills/`: Tooling for AST parsing (Tree-sitter) and File I/O.
- `src/pipeline/`: Core feedback loop logic.
- `src/utils/`: Telemetry and logging utilities.

## Getting Started

### Prerequisites
- Rust 1.83+
- OpenRouter API Key

### Build
```bash
cargo build
```

### Configuration
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_api_key_here
```
