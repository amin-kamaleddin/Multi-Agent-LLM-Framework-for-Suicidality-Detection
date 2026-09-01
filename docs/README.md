# single_agent_frameworks Documentation

This folder documents the current `single_agent_frameworks` implementation in this repository.

## Documents

- [DESIGN_SPEC.md](DESIGN_SPEC.md): architecture, system boundaries, components, data flow, persistence model, and extension points.
- [USAGE.md](USAGE.md): environment setup, command-line usage, output artifacts, retry workflow, and model-suite execution.
- [sample/suicide_detection/README.md](sample/suicide_detection/README.md): complete end-to-end suicide detection usage, including all agent modes and practical command recipes.

## Scope

These docs describe the code that currently exists under:

- [single_agent_frameworks/single_agent_frameworks](../single_agent_frameworks/single_agent_frameworks)
- [single_agent_frameworks/example/suicide_detection](../single_agent_frameworks/example/suicide_detection)

They are intended to be the authoritative reference for the present implementation, especially where the root [README.md](../README.md) is older than the codebase.

## Feature Highlights Covered Here

- Unified classification output handling in core framework:
	- string label outputs
	- structured class-score outputs
- Suicide detection modes:
	- `single-class`
	- `multi-class-chance`
	- `base-vs-rest-binary` with configurable `base_class`
- In-place retry workflow based on previous run artifacts.
