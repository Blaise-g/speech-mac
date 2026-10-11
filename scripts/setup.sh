#!/bin/sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
git config --local core.hooksPath .githooks
printf 'Installed repository-local hooks from %s/.githooks\n' "$repo_root"
