#!/bin/bash
set -ex

# Get the absolute current dir where the script is located
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure CURDIR is valid
if [ ! -d "$CURDIR" ]; then
    echo "Error: CURDIR '$CURDIR' is not a valid directory" >&2
    exit 1
fi

echo "Running from: $CURDIR"

# Check for binary existence helper
check_binary() {
    local binary="$1"
    if [ -x "$CURDIR/$binary" ]; then
        echo "$binary"
        return 0
    fi
    return 1
}

# Detect CPU features
BINARY=""

# Try fallback first (always available if built)
if check_binary "ikllama-cpp-fallback"; then
    BINARY="ikllama-cpp-fallback"
fi

# AVX?
if grep -qE '\savx\s' /proc/cpuinfo; then
    echo "CPU: AVX found"
    if check_binary "ikllama-cpp-avx"; then
        BINARY="ikllama-cpp-avx"
    fi
fi

# AVX2?
if grep -qE '\savx2\s' /proc/cpuinfo; then
    echo "CPU: AVX2 found"
    if check_binary "ikllama-cpp-avx2"; then
        BINARY="ikllama-cpp-avx2"
    fi
fi

# AVX-512?
if grep -qE '\savx512f\s' /proc/cpuinfo; then
    echo "CPU: AVX512F found"
    if check_binary "ikllama-cpp-avx512"; then
        BINARY="ikllama-cpp-avx512"
    fi
fi

# gRPC mode?
if [ -n "${LLAMACPP_GRPC_SERVERS:-}" ] && check_binary "ikllama-cpp-grpc"; then
    BINARY="ikllama-cpp-grpc"
fi

# Fallback if still empty
if [ -z "$BINARY" ]; then
    echo "Error: No ikllama-cpp binary found. Expected at least one of:" >&2
    echo "  - ikllama-cpp-fallback" >&2
    echo "  - ikllama-cpp-avx" >&2
    echo "  - ikllama-cpp-avx2" >&2
    echo "  - ikllama-cpp-avx512" >&2
    echo "  - ikllama-cpp-grpc" >&2
    echo "Current directory: $CURDIR" >&2
    ls -la "$CURDIR"/ikllama-cpp* 2>/dev/null || echo "No ikllama-cpp binaries found" >&2
    exit 1
fi

echo "Using binary: $BINARY"

# Setup library path
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="$CURDIR/lib:${DYLD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$CURDIR/lib:${LD_LIBRARY_PATH:-}"
fi

# Optional: use custom ld.so if present (e.g., for static linking or glibc compatibility)
if [ -f "$CURDIR/lib/ld.so" ]; then
    echo "Using custom ld.so: $CURDIR/lib/ld.so"
    exec "$CURDIR/lib/ld.so" "$CURDIR/$BINARY" "$@"
fi

# Direct execution
exec "$CURDIR/$BINARY" "$@"
