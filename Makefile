.DEFAULT_GOAL := help

CLANG ?= xcrun clang
CLANG_FORMAT ?= xcrun clang-format
BUILD_DIR ?= .build
BASE_CFLAGS ?= -std=c17 -Wall -Wextra -Wpedantic -Werror
OPT_CFLAGS ?= -O2
TEST_CFLAGS ?= -O0 -g
CPPFLAGS ?= -Isrc
LDLIBS ?= -framework Accelerate

APP_SOURCES := $(shell find src -type f -name '*.c' -print 2>/dev/null | sort)
MAIN_SOURCES := $(filter %/main.c,$(APP_SOURCES))
TEST_SOURCES := $(shell find tests -type f -name 'test_*.c' -print 2>/dev/null | sort)
ENGINE_SOURCES := $(filter-out %/main.c,$(APP_SOURCES))
C_SOURCES := $(shell find src tests benchmarks -type f -name '*.c' -print 2>/dev/null | sort)
FORMAT_FILES := $(shell find src tests benchmarks -type f \( -name '*.c' -o -name '*.h' \) -print 2>/dev/null | sort)

.PHONY: help setup dev format format-check lint typecheck build test test-one check-fast check clean source-discovery benchmark-build benchmark

help:
	@printf '%s\n' \
	  'make setup                         install the repository-local pre-commit hook' \
	  'make dev                           build the development CLI' \
	  'make format                        format C sources and headers' \
	  'make test-one TEST=tests/test_x.c   build and run one focused test' \
	  'make test                          build and run the full C test suite' \
	  'make check-fast                    format check, lint, typecheck, and tests' \
	  'make check                         run the full local check sequence' \
	  'make benchmark                     run the repeated native baseline (release -O2)' \
	  'make clean                         remove local build artifacts'

setup:
	@./scripts/setup.sh

dev:
	@$(MAKE) build OPT_CFLAGS='-O0 -g'

format: source-discovery
	@if [ -z "$(FORMAT_FILES)" ]; then \
	  echo 'No C sources or headers to format yet.'; \
	else \
	  $(CLANG_FORMAT) -i $(FORMAT_FILES); \
	fi

format-check: source-discovery
	@if [ -z "$(FORMAT_FILES)" ]; then \
	  echo 'Format check skipped: no C sources or headers yet.'; \
	else \
	  $(CLANG_FORMAT) --dry-run --Werror $(FORMAT_FILES); \
	fi

lint: source-discovery
	@if [ -z "$(C_SOURCES)" ]; then \
	  echo 'Lint skipped: no C sources yet.'; \
	else \
	  set -e; \
	  for source in $(C_SOURCES); do \
	    $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) --analyze -Xanalyzer -analyzer-output=text "$$source"; \
	  done; \
	fi

typecheck: source-discovery
	@if [ -z "$(C_SOURCES)" ]; then \
	  echo 'Typecheck skipped: no C sources yet.'; \
	else \
	  $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) -fsyntax-only $(C_SOURCES); \
	fi

build: source-discovery
	@if [ -z "$(APP_SOURCES)" ]; then \
	  echo 'Build skipped: no runtime C sources yet.'; \
	elif [ -z "$(MAIN_SOURCES)" ]; then \
	  mkdir -p $(BUILD_DIR)/obj; \
	  set -e; \
	  for source in $(APP_SOURCES); do \
	    object=$(BUILD_DIR)/obj/$$(basename "$$source" .c).o; \
	    $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(OPT_CFLAGS) -c "$$source" -o "$$object"; \
	  done; \
	  ar rcs $(BUILD_DIR)/libkokoro.a $(BUILD_DIR)/obj/*.o; \
	  echo 'Built .build/libkokoro.a; the CLI entrypoint is scheduled for a later ticket.'; \
	else \
	  mkdir -p $(BUILD_DIR); \
	  $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(OPT_CFLAGS) $(APP_SOURCES) $(LDLIBS) -o $(BUILD_DIR)/kokoro; \
	fi

benchmark-build: source-discovery
	@$(MAKE) build OPT_CFLAGS='-O2'
	@mkdir -p $(BUILD_DIR)
	@$(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) -O2 -DKOKORO_BENCHMARK_TIMING benchmarks/native_driver.c $(ENGINE_SOURCES) $(LDLIBS) -o $(BUILD_DIR)/kokoro-benchmark

benchmark: benchmark-build
	@UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked --no-sync python benchmarks/run.py --runs 5 --warm-repetitions 5 --output artifacts/12-performance-baseline.json

benchmark: export KOKORO_BENCH_COMPILER = $(CLANG)
benchmark: export KOKORO_BENCH_CPPFLAGS = $(CPPFLAGS)
benchmark: export KOKORO_BENCH_BASE_CFLAGS = $(BASE_CFLAGS)
benchmark: export KOKORO_BENCH_OPT_CFLAGS = -O2
benchmark: export KOKORO_BENCH_LDLIBS = $(LDLIBS)
benchmark: export KOKORO_BENCH_BUILD_DIR = $(BUILD_DIR)

test: source-discovery
	@if [ -z "$(TEST_SOURCES)" ]; then \
	  echo 'Test suite empty: no tests/test_*.c files yet.'; \
	else \
	  set -e; \
	  mkdir -p $(BUILD_DIR)/tests; \
	  for source in $(TEST_SOURCES); do \
	    name=$$(basename "$$source" .c); \
	    binary=$(BUILD_DIR)/tests/$$name; \
	    $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(TEST_CFLAGS) "$$source" $(ENGINE_SOURCES) $(LDLIBS) -o "$$binary"; \
	    "$$binary"; \
	  done; \
	fi

test-one: source-discovery
	@if [ -z "$(TEST)" ]; then \
	  echo 'Set TEST to one tests/test_*.c file.' >&2; \
	  exit 2; \
	fi
	@set -eu; \
	  repo_root=$$(pwd -P); \
	  test_path="$(TEST)"; \
	  test_dir=$$(cd "$$(dirname "$$test_path")" && pwd -P); \
	  test_name=$$(basename "$$test_path"); \
	  if [ "$$test_dir" != "$$repo_root/tests" ] || [ -L "$$test_dir/$$test_name" ] || [ ! -f "$$test_dir/$$test_name" ]; then \
	    echo 'Focused test must be a regular file directly inside tests/.' >&2; \
	    exit 2; \
	  fi; \
	  case "$$test_name" in test_*.c) ;; *) echo 'Focused tests must use the tests/test_*.c name.' >&2; exit 2 ;; esac; \
	  mkdir -p $(BUILD_DIR)/tests; \
	  name=$${test_name%.c}; \
	  binary=$(BUILD_DIR)/tests/$$name; \
	  $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(TEST_CFLAGS) "$$test_dir/$$test_name" $(ENGINE_SOURCES) $(LDLIBS) -o "$$binary"; \
	  "$$binary"

source-discovery:
	@find src -type f \( -name '*.c' -o -name '*.h' \) -print >/dev/null
	@find tests -type f \( -name '*.c' -o -name '*.h' \) -print >/dev/null
	@find benchmarks -type f \( -name '*.c' -o -name '*.h' \) -print >/dev/null

check-fast: format-check lint typecheck test

check: format-check lint typecheck build test

clean:
	rm -rf $(BUILD_DIR)
