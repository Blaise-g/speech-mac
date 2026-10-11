# Coding standards

- Write runtime inference in readable C17 and use the repository `.clang-format` rules.
- Keep ownership and lifetime visible at interfaces. Document borrowed data and transfers of ownership where they are not obvious from the function contract.
- Keep platform calls behind narrow interfaces when that makes CPU reference checks possible without changing inference behavior.
- Add runtime dependencies only when they serve an accepted requirement; Python remains a reference and conversion tool.

The compiler flags, formatter, static analyzer, and check order are defined in the Makefile.
