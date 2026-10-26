.DEFAULT_GOAL := help

CLANG ?= xcrun clang
CLANG_FORMAT ?= xcrun clang-format
BUILD_DIR ?= .build
BASE_CFLAGS ?= -std=c17 -Wall -Wextra -Wpedantic -Werror
OPT_CFLAGS ?= -O2
TEST_CFLAGS ?= -O0 -g
CPPFLAGS ?= -Isrc
LDLIBS ?= -framework AudioToolbox -framework Accelerate

APP_SOURCES := $(shell find src -type f -name '*.c' -print 2>/dev/null | sort)
TEST_SOURCES := $(shell find tests -type f -name 'test_*.c' -print 2>/dev/null | sort)
ENGINE_SOURCES := $(filter-out %/speech_main.c,$(APP_SOURCES))
SPEECH_SOURCES := src/speech_main.c $(ENGINE_SOURCES)
C_SOURCES := $(shell find src tests benchmarks -type f -name '*.c' -print 2>/dev/null | sort)
FORMAT_FILES := $(shell find src tests benchmarks -type f \( -name '*.c' -o -name '*.h' \) -print 2>/dev/null | sort)

.PHONY: help bootstrap bootstrap-test setup dev format format-check lint typecheck build test test-one check-fast check clean source-discovery phonon-reference-check phonon-native-pack phonon-native-check english-normalization-probe

help:
	@printf '%s\n' \
	  'make bootstrap                     prepare verified assets and build the Mac CLI' \
	  'make setup                         install the repository-local pre-commit hook' \
	  'make dev                           build the development CLI' \
	  'make format                        format C sources and headers' \
	  'make test-one TEST=tests/test_x.c   build and run one focused test' \
	  'make test                          build and run the full C test suite' \
	  'make check-fast                    format check, lint, typecheck, and tests' \
	  'make check                         run the full local check sequence' \
	  'make clean                         remove local build artifacts'

bootstrap:
	@sh scripts/bootstrap.sh

bootstrap-test:
	@uv run --no-project --python 3.12 python -m unittest discover -s scripts/tests

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
	@mkdir -p $(BUILD_DIR)
	@$(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(OPT_CFLAGS) $(SPEECH_SOURCES) $(LDLIBS) -o $(BUILD_DIR)/speech-mac

test: source-discovery
	@if [ -z "$(TEST_SOURCES)" ]; then \
	  echo 'Test suite empty: no tests/test_*.c files yet.'; \
	else \
	  set -e; \
	  mkdir -p $(BUILD_DIR)/tests; \
	  for source in $(TEST_SOURCES); do \
	    name=$$(basename "$$source" .c); \
	    binary=$(BUILD_DIR)/tests/$$name; \
	    $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(TEST_CFLAGS) -DPHONON_PACK_TEST_API "$$source" $(ENGINE_SOURCES) $(LDLIBS) -o "$$binary"; \
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
	  $(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) $(TEST_CFLAGS) -DPHONON_PACK_TEST_API "$$test_dir/$$test_name" $(ENGINE_SOURCES) $(LDLIBS) -o "$$binary"; \
	  "$$binary"

source-discovery:
	@find src -type f \( -name '*.c' -o -name '*.h' \) -print >/dev/null
	@find tests -type f \( -name '*.c' -o -name '*.h' \) -print >/dev/null
	@find benchmarks -type f \( -name '*.c' -o -name '*.h' \) -print >/dev/null

check-fast: format-check lint typecheck test

check: format-check lint typecheck build test removal-check bootstrap-test

clean:
	rm -rf $(BUILD_DIR)

phonon-reference-check:
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/verify_assets.py
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/oracle.py verify
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/oracle.py verify-mimi
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/oracle.py verify-temporal
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/oracle.py verify-q8
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests

phonon-native-pack:
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/native_pack.py
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/native_projection_oracle.py

phonon-native-check: phonon-native-pack
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/export_flow_fixture.py
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/export_sampler_fixture.py
	@PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack \
	 PHONON_FLOW_SAMPLER_FIXTURE=.build/phonon-flow-sampler-v1.bin $(MAKE) test-one TEST=tests/test_phonon_transformer.c
	@UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/verify_native_generation_trace.py
	@PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack \
	 PHONON_Q8_ORACLE=.build/phonon-native-q8-projections-v1.bin \
	 PHONON_BF16_ORACLE=.build/phonon-native-bf16-oracle-v1.bin $(MAKE) test-one TEST=tests/test_phonon_pack.c


english-normalization-probe: source-discovery
	@mkdir -p $(BUILD_DIR)
	@$(CLANG) $(CPPFLAGS) $(BASE_CFLAGS) -O2 tests/english_normalize_probe.c src/english_normalize.c src/utf8.c src/speech_error.c -o $(BUILD_DIR)/english-normalize-probe

.PHONY: removal-check
removal-check:
	@uv run --no-project --python 3.12 python -m unittest discover -s reference/phonon/tests -p test_maintained_surface.py
