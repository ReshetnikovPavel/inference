#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

header=/usr/include/level_zero/ze_api.h
[[ -f "$header" ]] || {
    echo "missing $header; install level-zero-headers" >&2
    exit 1
}
command -v bindgen >/dev/null || {
    echo "bindgen not found; run scripts/setup.sh or install rust-bindgen" >&2
    exit 1
}

api_version=$(
    sed -n 's/^#define ZE_API_VERSION_CURRENT_M[[:space:]]*ZE_MAKE_VERSION([[:space:]]*\([0-9][0-9]*\),[[:space:]]*\([0-9][0-9]*\).*/\1.\2/p' "$header"
)
if [[ -z "$api_version" ]]; then
    api_version=$(
        sed -n 's/^[[:space:]]*ZE_API_VERSION_CURRENT[[:space:]]*=[[:space:]]*ZE_MAKE_VERSION([[:space:]]*\([0-9][0-9]*\),[[:space:]]*\([0-9][0-9]*\).*/\1.\2/p' "$header"
    )
fi
[[ -n "$api_version" ]] || api_version=unknown

mkdir -p src/l0

bindgen "$header" \
    -o src/l0/sys.rs \
    --allowlist-function '^ze[A-Z].*' \
    --allowlist-type '^ze_.*_t$' \
    --allowlist-var '^ZE_.*' \
    --default-enum-style consts \
    --no-prepend-enum-name \
    --raw-line "// Generated from $header (Level Zero API $api_version). Do not edit."

echo "wrote src/l0/sys.rs from $header (API $api_version)"
