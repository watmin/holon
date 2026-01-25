# Architecture Decision Records (ADRs)

This directory contains the key architectural decisions made during Holon's development.

## ADR Format

Each ADR follows this structure:
- **Status**: Accepted/Rejected/Deprecated
- **Context**: Problem statement and constraints
- **Decision**: Chosen solution and rationale
- **Consequences**: Benefits, drawbacks, and mitigations

## Key Decisions

### Core Architecture
- **[001: VSA/HDC Architecture](001-vsa-hdc-architecture.md)** - Foundational vector symbolic approach
- **[004: ANN Indexing Strategy](004-ann-indexing-strategy.md)** - Performance optimization with accuracy guarantees

### Data & Query Systems
- **[002: Data Format Support](002-data-format-support.md)** - JSON/EDN dual format implementation
- **[003: Query System Design](003-query-system-design.md)** - Guards, negations, and $or implementation

## Decision Principles

### Data over Assumptions
Architecture favors explicit data modeling over implicit assumptions. All major design choices are:
- **Tested**: Comprehensive test coverage validates decisions
- **Measured**: Performance benchmarks quantify trade-offs
- **Documented**: Clear rationale prevents future technical debt

### Progressive Enhancement
Start simple, add complexity only when justified by real use cases and performance requirements.

### Research-Driven
Many decisions stem from cognitive science research (VSA/HDC) while maintaining practical engineering constraints.

## Contributing

When proposing architectural changes:
1. Create new ADR in this directory (numbered sequentially)
2. Follow the established format
3. Include performance implications
4. Reference supporting data/tests
5. Update this README

## References

- [Architectural Context](../holon_context.md)
- [Performance Benchmarks](../performance.md)
- [Challenge Solutions](../../scripts/challenges/)
