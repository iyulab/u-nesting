//! # U-Nesting FFI
//!
//! C FFI interface for the U-Nesting spatial optimization engine.
//!
//! This crate provides a C-compatible interface for using U-Nesting from
//! other languages like C#, Python, etc.
//!
//! ## Functions
//!
//! ### Basic API
//! - [`unesting_solve`] - Auto-detects 2D/3D mode and solves
//! - [`unesting_solve_2d`] - Solves 2D nesting problems
//! - [`unesting_solve_3d`] - Solves 3D bin packing problems
//! - [`unesting_free_string`] - Frees result strings
//! - [`unesting_version`] - Returns API version
//!
//! ### Progress Callback API
//! - [`unesting_solve_with_progress`] - Auto-detects mode with progress callback
//! - [`unesting_solve_2d_with_progress`] - 2D nesting with progress callback
//! - [`unesting_solve_3d_with_progress`] - 3D packing with progress callback
//!
//! ## Error Codes
//!
//! | Code | Constant | Meaning |
//! |------|----------|---------|
//! | 0 | `UNESTING_OK` | Success |
//! | -1 | `UNESTING_ERR_NULL_PTR` | Null pointer passed |
//! | -2 | `UNESTING_ERR_INVALID_JSON` | Invalid JSON input |
//! | -3 | `UNESTING_ERR_SOLVE_FAILED` | Solver failed |
//! | -4 | `UNESTING_ERR_CANCELLED` | Cancelled by callback |
//! | -99 | `UNESTING_ERR_UNKNOWN` | Unknown error |
//!
//! ## Progress Callback
//!
//! The progress callback receives JSON with these fields:
//!
//! ```json
//! {
//!   "iteration": 10,
//!   "total_iterations": 100,
//!   "utilization": 0.75,
//!   "best_fitness": 1.5,
//!   "items_placed": 5,
//!   "total_items": 10,
//!   "elapsed_ms": 500,
//!   "phase": "Optimizing",
//!   "running": true
//! }
//! ```
//!
//! Return non-zero from callback to continue, zero to cancel.
//!
//! ## JSON Request Format (2D)
//!
//! ```json
//! {
//!   "mode": "2d",
//!   "geometries": [
//!     {
//!       "id": "part1",
//!       "polygon": [[0,0], [100,0], [100,50], [0,50]],
//!       "quantity": 5,
//!       "rotations": [0, 90, 180, 270],
//!       "allow_flip": false
//!     }
//!   ],
//!   "boundary": {
//!     "width": 1000,
//!     "height": 500
//!   },
//!   "config": {
//!     "strategy": "nfp",
//!     "spacing": 2.0,
//!     "margin": 5.0,
//!     "time_limit_ms": 30000
//!   }
//! }
//! ```
//!
//! ## JSON Request Format (3D)
//!
//! ```json
//! {
//!   "mode": "3d",
//!   "geometries": [
//!     {
//!       "id": "box1",
//!       "dimensions": [100, 50, 30],
//!       "quantity": 10,
//!       "mass": 2.5
//!     }
//!   ],
//!   "boundary": {
//!     "dimensions": [500, 400, 300],
//!     "max_mass": 100.0,
//!     "gravity": true,
//!     "stability": true
//!   },
//!   "config": {
//!     "strategy": "ep",
//!     "time_limit_ms": 30000
//!   }
//! }
//! ```
//!
//! ## Strategy Options
//!
//! | Strategy | 2D | 3D | Description |
//! |----------|----|----|-------------|
//! | `blf` | ✓ | ✓ | Bottom-Left Fill (fast) |
//! | `nfp` | ✓ | - | NFP-guided placement |
//! | `ga` | ✓ | ✓ | Genetic Algorithm |
//! | `brkga` | ✓ | ✓ | Biased Random-Key GA |
//! | `sa` | ✓ | ✓ | Simulated Annealing |
//! | `ep` | - | ✓ | Extreme Point heuristic |
//!
//! ## C Example (with Progress Callback)
//!
//! ```c
//! #include "unesting.h"
//! #include <stdio.h>
//!
//! int progress_callback(const char* json, void* user_data) {
//!     printf("Progress: %s\n", json);
//!     return 1; // Continue (return 0 to cancel)
//! }
//!
//! int main() {
//!     const char* request = "{\"geometries\": [...], \"boundary\": {...}}";
//!     char* result = NULL;
//!     int code = unesting_solve_2d_with_progress(request, progress_callback, NULL, &result);
//!     if (code == UNESTING_OK) {
//!         printf("Result: %s\n", result);
//!     }
//!     unesting_free_string(result);
//!     return code;
//! }
//! ```
//!
//! ## C# Example
//!
//! ```csharp
//! [DllImport("u_nesting_ffi")]
//! static extern int unesting_solve_2d(string json, out IntPtr result);
//!
//! [DllImport("u_nesting_ffi")]
//! static extern void unesting_free_string(IntPtr ptr);
//!
//! string json = "{\"geometries\": [...], \"boundary\": {...}}";
//! IntPtr resultPtr;
//! int code = unesting_solve_2d(json, out resultPtr);
//! string result = Marshal.PtrToStringAnsi(resultPtr);
//! unesting_free_string(resultPtr);
//! ```

mod api;
mod callback;
mod types;

pub use api::*;
pub use callback::*;
pub use types::*;
