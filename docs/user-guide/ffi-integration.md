# FFI Integration Guide

This guide covers integrating U-Nesting into applications via the C FFI interface.

## Overview

U-Nesting provides a C-compatible Foreign Function Interface (FFI) that enables integration with:

- C/C++ applications
- C# via P/Invoke
- Python via ctypes/cffi
- Java via JNI/JNA
- Go via cgo
- Any language with C FFI support

## Building the FFI Library

```bash
# Release build
cargo build -p u-nesting-ffi --release

# Output locations:
# Windows: target/release/u_nesting_ffi.dll
# Linux:   target/release/libu_nesting_ffi.so
# macOS:   target/release/libu_nesting_ffi.dylib
```

## C Header

The header is auto-generated during build:

```bash
# Generate header manually
cargo build -p u-nesting-ffi
# Output: target/u_nesting.h
```

### Header Contents

```c
#ifndef U_NESTING_H
#define U_NESTING_H

#include <stdint.h>

/* Error codes */
#define UNESTING_OK             0
#define UNESTING_ERR_NULL_PTR   -1
#define UNESTING_ERR_INVALID_JSON -2
#define UNESTING_ERR_SOLVE_FAILED -3
#define UNESTING_ERR_CANCELLED  -4
#define UNESTING_ERR_UNKNOWN    -99

/* Progress callback type */
typedef int (*UnestingProgressCallback)(const char* progress_json, void* user_data);

/* Basic API */
int unesting_solve(const char* request_json, char** result_ptr);
int unesting_solve_2d(const char* request_json, char** result_ptr);
int unesting_solve_3d(const char* request_json, char** result_ptr);

/* Progress API */
int unesting_solve_with_progress(
    const char* request_json,
    UnestingProgressCallback callback,
    void* user_data,
    char** result_ptr
);

int unesting_solve_2d_with_progress(
    const char* request_json,
    UnestingProgressCallback callback,
    void* user_data,
    char** result_ptr
);

int unesting_solve_3d_with_progress(
    const char* request_json,
    UnestingProgressCallback callback,
    void* user_data,
    char** result_ptr
);

/* Utility */
void unesting_free_string(char* ptr);
const char* unesting_version(void);

#endif /* U_NESTING_H */
```

## API Reference

### unesting_solve

Auto-detects 2D/3D mode based on request JSON.

```c
int unesting_solve(
    const char* request_json,  // Input: JSON request string
    char** result_ptr          // Output: Pointer to result JSON string
);
```

**Returns**: Error code (0 = success)

### unesting_solve_2d / unesting_solve_3d

Explicit 2D nesting or 3D packing.

```c
int unesting_solve_2d(const char* request_json, char** result_ptr);
int unesting_solve_3d(const char* request_json, char** result_ptr);
```

### unesting_solve_*_with_progress

Solving with progress callback:

```c
int unesting_solve_2d_with_progress(
    const char* request_json,
    UnestingProgressCallback callback,  // Can be NULL
    void* user_data,                    // Passed to callback
    char** result_ptr
);
```

**Callback signature**:
```c
int callback(const char* progress_json, void* user_data);
// Return: non-zero to continue, 0 to cancel
```

### unesting_free_string

Frees memory allocated by the library:

```c
void unesting_free_string(char* ptr);
```

**Important**: Always call this for result strings to prevent memory leaks.

### unesting_version

Returns the library version:

```c
const char* unesting_version(void);
// Returns: "0.1.0" (static string, do not free)
```

## Error Codes

| Code | Constant | Description |
|------|----------|-------------|
| 0 | `UNESTING_OK` | Success |
| -1 | `UNESTING_ERR_NULL_PTR` | Null pointer passed |
| -2 | `UNESTING_ERR_INVALID_JSON` | JSON parsing failed |
| -3 | `UNESTING_ERR_SOLVE_FAILED` | Solver error |
| -4 | `UNESTING_ERR_CANCELLED` | Cancelled by callback |
| -99 | `UNESTING_ERR_UNKNOWN` | Unknown error |

## Progress JSON Format

```json
{
  "iteration": 42,
  "total_iterations": 100,
  "utilization": 0.756,
  "best_fitness": 1.32,
  "items_placed": 15,
  "total_items": 20,
  "elapsed_ms": 1500,
  "phase": "Optimizing",
  "running": true
}
```

## C Example

### Basic Usage

```c
#include <stdio.h>
#include <stdlib.h>
#include "u_nesting.h"

int main() {
    const char* request = "{"
        "\"mode\": \"2d\","
        "\"geometries\": ["
        "  {\"id\": \"rect\", \"polygon\": [[0,0],[100,0],[100,50],[0,50]], \"quantity\": 5}"
        "],"
        "\"boundary\": {\"width\": 500, \"height\": 300}"
    "}";

    char* result = NULL;
    int code = unesting_solve(request, &result);

    if (code == UNESTING_OK) {
        printf("Result: %s\n", result);
    } else {
        printf("Error: %d\n", code);
    }

    if (result) {
        unesting_free_string(result);
    }

    return code;
}
```

### With Progress Callback

```c
#include <stdio.h>
#include "u_nesting.h"

typedef struct {
    int call_count;
    int max_calls;
} UserData;

int progress_callback(const char* json, void* user_data) {
    UserData* data = (UserData*)user_data;
    data->call_count++;

    printf("Progress: %s\n", json);

    // Cancel after max_calls
    if (data->max_calls > 0 && data->call_count >= data->max_calls) {
        return 0;  // Cancel
    }
    return 1;  // Continue
}

int main() {
    const char* request = "{...}";
    char* result = NULL;

    UserData data = { .call_count = 0, .max_calls = 0 };

    int code = unesting_solve_2d_with_progress(
        request,
        progress_callback,
        &data,
        &result
    );

    printf("Callback invoked %d times\n", data.call_count);

    if (code == UNESTING_ERR_CANCELLED) {
        printf("Operation was cancelled\n");
    }

    if (result) {
        unesting_free_string(result);
    }

    return 0;
}
```

## C++ Integration

```cpp
#include <string>
#include <stdexcept>
#include <functional>
#include "u_nesting.h"

class UNesting {
public:
    using ProgressCallback = std::function<bool(const std::string&)>;

    static std::string solve2D(const std::string& request) {
        char* result = nullptr;
        int code = unesting_solve_2d(request.c_str(), &result);

        if (code != UNESTING_OK) {
            throw std::runtime_error("Solve failed: " + std::to_string(code));
        }

        std::string json(result);
        unesting_free_string(result);
        return json;
    }

    static std::string solve2DWithProgress(
        const std::string& request,
        ProgressCallback callback
    ) {
        char* result = nullptr;

        auto wrapper = [](const char* json, void* data) -> int {
            auto* cb = static_cast<ProgressCallback*>(data);
            return (*cb)(json) ? 1 : 0;
        };

        int code = unesting_solve_2d_with_progress(
            request.c_str(),
            wrapper,
            &callback,
            &result
        );

        if (code == UNESTING_ERR_CANCELLED) {
            throw std::runtime_error("Cancelled");
        }
        if (code != UNESTING_OK) {
            throw std::runtime_error("Solve failed: " + std::to_string(code));
        }

        std::string json(result);
        unesting_free_string(result);
        return json;
    }
};
```

## Thread Safety

The FFI functions are **thread-safe**:

- Multiple threads can call solving functions simultaneously
- Each call has its own isolated state
- Callbacks are invoked on the calling thread

**Important**: Progress callbacks must be thread-safe if shared state is accessed.

## Memory Management

### Rules

1. **Input strings**: Caller owns, library reads only
2. **Output strings**: Library allocates, caller must free with `unesting_free_string`
3. **Version string**: Static, do not free

### Example

```c
char* result = NULL;
int code = unesting_solve(request, &result);

// Process result...

// Always free, even if code != UNESTING_OK
if (result) {
    unesting_free_string(result);
    result = NULL;  // Good practice
}
```

## Platform-Specific Notes

### Windows

```c
// Link against u_nesting_ffi.lib (import library)
// Runtime requires u_nesting_ffi.dll in PATH or same directory
#pragma comment(lib, "u_nesting_ffi.lib")
```

### Linux

```bash
# Compile
gcc -o app main.c -L/path/to -lu_nesting_ffi

# Run (set library path)
LD_LIBRARY_PATH=/path/to ./app
```

### macOS

```bash
# Compile
clang -o app main.c -L/path/to -lu_nesting_ffi

# Run
DYLD_LIBRARY_PATH=/path/to ./app
```

## Troubleshooting

### Library Not Found

```
error: cannot find -lu_nesting_ffi
```

**Solution**: Add library path with `-L/path/to/lib`

### Symbol Not Found

```
undefined symbol: unesting_solve
```

**Solution**: Ensure library is built with `cdecl` calling convention (default on Unix).

### Memory Corruption

**Symptoms**: Crashes, garbled output

**Causes**:
- Not freeing result strings
- Using result after free
- Passing invalid JSON

**Solution**: Use memory sanitizers during development:

```bash
RUSTFLAGS="-Z sanitizer=address" cargo build -p u-nesting-ffi
```
