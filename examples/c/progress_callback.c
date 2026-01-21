/**
 * @file progress_callback.c
 * @brief Example demonstrating U-Nesting FFI with progress callback
 *
 * This example shows how to use the U-Nesting library from C with
 * real-time progress reporting via callback function.
 *
 * Build:
 *   gcc -o progress_callback progress_callback.c -L../../target/release -lu_nesting_ffi
 *
 * Run (Windows):
 *   set PATH=%PATH%;../../target/release && progress_callback.exe
 *
 * Run (Linux/macOS):
 *   LD_LIBRARY_PATH=../../target/release ./progress_callback
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations for FFI functions */
typedef int (*UnestingProgressCallback)(const char* progress_json, void* user_data);

extern int unesting_solve_2d_with_progress(
    const char* request_json,
    UnestingProgressCallback callback,
    void* user_data,
    char** result_ptr
);

extern int unesting_solve_3d_with_progress(
    const char* request_json,
    UnestingProgressCallback callback,
    void* user_data,
    char** result_ptr
);

extern void unesting_free_string(char* ptr);
extern const char* unesting_version(void);

/* Error codes */
#define UNESTING_OK             0
#define UNESTING_ERR_NULL_PTR   -1
#define UNESTING_ERR_INVALID_JSON -2
#define UNESTING_ERR_SOLVE_FAILED -3
#define UNESTING_ERR_CANCELLED  -4
#define UNESTING_ERR_UNKNOWN    -99

/* User data structure for callback */
typedef struct {
    int call_count;
    int max_calls;       /* Max calls before cancelling (0 = no limit) */
    int verbose;         /* Print progress JSON */
} CallbackData;

/**
 * Progress callback function
 *
 * @param progress_json JSON string with progress information
 * @param user_data     Pointer to CallbackData struct
 * @return Non-zero to continue, zero to cancel
 */
int progress_callback(const char* progress_json, void* user_data) {
    CallbackData* data = (CallbackData*)user_data;
    data->call_count++;

    if (data->verbose) {
        printf("Progress [%d]: %s\n", data->call_count, progress_json);
    } else {
        printf(".");
        fflush(stdout);
    }

    /* Cancel if max_calls reached */
    if (data->max_calls > 0 && data->call_count >= data->max_calls) {
        printf("\nCancelling after %d callbacks\n", data->call_count);
        return 0; /* Cancel */
    }

    return 1; /* Continue */
}

/**
 * Run 2D nesting example
 */
void example_2d_nesting(void) {
    printf("\n=== 2D Nesting Example ===\n");

    const char* request = "{"
        "\"geometries\": ["
        "  {\"id\": \"rect1\", \"polygon\": [[0,0], [100,0], [100,50], [0,50]], \"quantity\": 5},"
        "  {\"id\": \"triangle\", \"polygon\": [[0,0], [80,0], [40,60]], \"quantity\": 3}"
        "],"
        "\"boundary\": {\"width\": 500, \"height\": 300},"
        "\"config\": {"
        "  \"strategy\": \"blf\","
        "  \"spacing\": 2.0"
        "}"
    "}";

    CallbackData data = { .call_count = 0, .max_calls = 0, .verbose = 1 };
    char* result = NULL;

    printf("Request: %s\n\n", request);

    int code = unesting_solve_2d_with_progress(request, progress_callback, &data, &result);

    printf("\n");
    printf("Return code: %d\n", code);
    printf("Callback invocations: %d\n", data.call_count);

    if (code == UNESTING_OK && result) {
        printf("Result: %s\n", result);
    } else if (code == UNESTING_ERR_CANCELLED) {
        printf("Solving was cancelled\n");
    } else {
        printf("Error occurred\n");
    }

    if (result) {
        unesting_free_string(result);
    }
}

/**
 * Run 3D bin packing example
 */
void example_3d_packing(void) {
    printf("\n=== 3D Bin Packing Example ===\n");

    const char* request = "{"
        "\"geometries\": ["
        "  {\"id\": \"small\", \"dimensions\": [20, 20, 20], \"quantity\": 10},"
        "  {\"id\": \"medium\", \"dimensions\": [40, 30, 25], \"quantity\": 5},"
        "  {\"id\": \"large\", \"dimensions\": [60, 40, 30], \"quantity\": 3}"
        "],"
        "\"boundary\": {"
        "  \"dimensions\": [200, 150, 100],"
        "  \"gravity\": true,"
        "  \"stability\": true"
        "},"
        "\"config\": {"
        "  \"strategy\": \"ep\""
        "}"
    "}";

    CallbackData data = { .call_count = 0, .max_calls = 0, .verbose = 1 };
    char* result = NULL;

    printf("Request: %s\n\n", request);

    int code = unesting_solve_3d_with_progress(request, progress_callback, &data, &result);

    printf("\n");
    printf("Return code: %d\n", code);
    printf("Callback invocations: %d\n", data.call_count);

    if (code == UNESTING_OK && result) {
        printf("Result: %s\n", result);
    }

    if (result) {
        unesting_free_string(result);
    }
}

/**
 * Demonstrate cancellation
 */
void example_cancellation(void) {
    printf("\n=== Cancellation Example ===\n");

    const char* request = "{"
        "\"geometries\": ["
        "  {\"id\": \"rect\", \"polygon\": [[0,0], [10,0], [10,5], [0,5]], \"quantity\": 50}"
        "],"
        "\"boundary\": {\"width\": 500, \"height\": 500},"
        "\"config\": {\"strategy\": \"blf\"}"
    "}";

    /* Cancel after first callback */
    CallbackData data = { .call_count = 0, .max_calls = 1, .verbose = 1 };
    char* result = NULL;

    int code = unesting_solve_2d_with_progress(request, progress_callback, &data, &result);

    printf("Return code: %d (expected: %d CANCELLED)\n", code, UNESTING_ERR_CANCELLED);

    if (result) {
        printf("Result: %s\n", result);
        unesting_free_string(result);
    }
}

/**
 * No callback (NULL) example
 */
void example_no_callback(void) {
    printf("\n=== No Callback Example ===\n");

    const char* request = "{"
        "\"geometries\": ["
        "  {\"id\": \"rect\", \"polygon\": [[0,0], [10,0], [10,5], [0,5]], \"quantity\": 3}"
        "],"
        "\"boundary\": {\"width\": 100, \"height\": 100}"
    "}";

    char* result = NULL;

    /* Pass NULL for callback - no progress reporting */
    int code = unesting_solve_2d_with_progress(request, NULL, NULL, &result);

    printf("Return code: %d\n", code);

    if (code == UNESTING_OK && result) {
        printf("Result: %s\n", result);
    }

    if (result) {
        unesting_free_string(result);
    }
}

int main(int argc, char* argv[]) {
    printf("U-Nesting FFI Progress Callback Example\n");
    printf("Version: %s\n", unesting_version());
    printf("========================================\n");

    /* Run examples */
    example_2d_nesting();
    example_3d_packing();
    example_cancellation();
    example_no_callback();

    printf("\n=== All examples completed ===\n");
    return 0;
}
