//! FFI Callback types and wrappers.
//!
//! This module provides C ABI callback function pointer types for progress reporting
//! and thread-safe wrappers for calling them from Rust.

use serde::Serialize;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};

/// C ABI progress callback function type.
///
/// The callback receives a JSON string containing progress information.
/// The string is valid only for the duration of the callback call.
///
/// # Parameters
/// - `progress_json`: Null-terminated JSON string with progress data
/// - `user_data`: Opaque pointer passed to the solve function
///
/// # Returns
/// - `true` (non-zero): Continue solving
/// - `false` (zero): Cancel solving
///
/// # C Declaration
/// ```c
/// typedef int (*UnestingProgressCallback)(const char* progress_json, void* user_data);
/// ```
pub type UnestingProgressCallback = Option<unsafe extern "C" fn(*const c_char, *mut c_void) -> i32>;

/// Progress information sent to callbacks as JSON.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressJson {
    /// Current iteration/generation number.
    pub iteration: u32,
    /// Total expected iterations (0 if unknown).
    pub total_iterations: u32,
    /// Current best utilization (0.0 to 1.0).
    pub utilization: f64,
    /// Current best fitness value.
    pub best_fitness: f64,
    /// Number of items placed.
    pub items_placed: usize,
    /// Total number of items.
    pub total_items: usize,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Current phase/stage description.
    pub phase: String,
    /// Whether the solver is still running.
    pub running: bool,
}

impl From<u_nesting_core::solver::ProgressInfo> for ProgressJson {
    fn from(p: u_nesting_core::solver::ProgressInfo) -> Self {
        Self {
            iteration: p.iteration,
            total_iterations: p.total_iterations,
            utilization: p.utilization,
            best_fitness: p.best_fitness,
            items_placed: p.items_placed,
            total_items: p.total_items,
            elapsed_ms: p.elapsed_ms,
            phase: p.phase,
            running: p.running,
        }
    }
}

/// Thread-safe wrapper for C progress callbacks.
///
/// This wrapper handles:
/// - Panic safety at FFI boundary
/// - JSON serialization of progress info
/// - Cancellation tracking
pub struct CallbackWrapper {
    callback: UnestingProgressCallback,
    user_data: *mut c_void,
    cancelled: AtomicBool,
}

// SAFETY: CallbackWrapper is designed for use across threads.
// The C callback and user_data are provided by the caller who guarantees thread safety.
unsafe impl Send for CallbackWrapper {}
unsafe impl Sync for CallbackWrapper {}

impl CallbackWrapper {
    /// Creates a new callback wrapper.
    ///
    /// # Safety
    /// - `callback` must be a valid function pointer or None
    /// - `user_data` must remain valid for the lifetime of this wrapper
    pub unsafe fn new(callback: UnestingProgressCallback, user_data: *mut c_void) -> Self {
        Self {
            callback,
            user_data,
            cancelled: AtomicBool::new(false),
        }
    }

    /// Returns true if the callback has requested cancellation.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Invokes the callback with progress information.
    ///
    /// Returns `true` to continue, `false` to cancel.
    pub fn invoke(&self, progress: &ProgressJson) -> bool {
        if self.is_cancelled() {
            return false;
        }

        let Some(callback) = self.callback else {
            return true; // No callback, continue
        };

        // Serialize to JSON
        let json = match serde_json::to_string(progress) {
            Ok(s) => s,
            Err(_) => return true, // Serialization failed, but continue solving
        };

        let c_string = match CString::new(json) {
            Ok(s) => s,
            Err(_) => return true, // CString creation failed, continue
        };

        // Call the C callback with panic guard
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            callback(c_string.as_ptr(), self.user_data)
        }));

        match result {
            Ok(continue_flag) => {
                if continue_flag == 0 {
                    self.cancelled.store(true, Ordering::Relaxed);
                    false
                } else {
                    true
                }
            }
            Err(_) => {
                // Panic in callback - treat as cancellation for safety
                self.cancelled.store(true, Ordering::Relaxed);
                false
            }
        }
    }

    /// Creates a Rust closure for use with the core solver's progress callback.
    pub fn as_progress_fn(&self) -> impl Fn(u_nesting_core::solver::ProgressInfo) -> bool + '_ {
        move |progress| {
            let json: ProgressJson = progress.into();
            self.invoke(&json)
        }
    }
}

/// Default callback interval in milliseconds.
pub const DEFAULT_CALLBACK_INTERVAL_MS: u32 = 100;

/// Minimum callback interval in milliseconds.
pub const MIN_CALLBACK_INTERVAL_MS: u32 = 10;

/// Maximum callback interval in milliseconds.
pub const MAX_CALLBACK_INTERVAL_MS: u32 = 10000;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    #[test]
    fn test_progress_json_serialization() {
        let progress = ProgressJson {
            iteration: 10,
            total_iterations: 100,
            utilization: 0.75,
            best_fitness: 1.5,
            items_placed: 5,
            total_items: 10,
            elapsed_ms: 500,
            phase: "Optimizing".to_string(),
            running: true,
        };

        let json = serde_json::to_string(&progress).unwrap();
        assert!(json.contains("\"iteration\":10"));
        assert!(json.contains("\"utilization\":0.75"));
        assert!(json.contains("\"phase\":\"Optimizing\""));
    }

    #[test]
    fn test_callback_wrapper_no_callback() {
        let wrapper = unsafe { CallbackWrapper::new(None, std::ptr::null_mut()) };

        let progress = ProgressJson {
            iteration: 1,
            total_iterations: 10,
            utilization: 0.5,
            best_fitness: 1.0,
            items_placed: 5,
            total_items: 10,
            elapsed_ms: 100,
            phase: "Test".to_string(),
            running: true,
        };

        // Should return true (continue) when no callback
        assert!(wrapper.invoke(&progress));
        assert!(!wrapper.is_cancelled());
    }

    #[test]
    fn test_callback_wrapper_continue() {
        static CALL_COUNT: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn continue_callback(_json: *const c_char, _data: *mut c_void) -> i32 {
            CALL_COUNT.fetch_add(1, Ordering::Relaxed);
            1 // Continue
        }

        let wrapper = unsafe { CallbackWrapper::new(Some(continue_callback), std::ptr::null_mut()) };

        let progress = ProgressJson {
            iteration: 1,
            total_iterations: 10,
            utilization: 0.5,
            best_fitness: 1.0,
            items_placed: 5,
            total_items: 10,
            elapsed_ms: 100,
            phase: "Test".to_string(),
            running: true,
        };

        CALL_COUNT.store(0, Ordering::Relaxed);
        assert!(wrapper.invoke(&progress));
        assert_eq!(CALL_COUNT.load(Ordering::Relaxed), 1);
        assert!(!wrapper.is_cancelled());
    }

    #[test]
    fn test_callback_wrapper_cancel() {
        unsafe extern "C" fn cancel_callback(_json: *const c_char, _data: *mut c_void) -> i32 {
            0 // Cancel
        }

        let wrapper = unsafe { CallbackWrapper::new(Some(cancel_callback), std::ptr::null_mut()) };

        let progress = ProgressJson {
            iteration: 1,
            total_iterations: 10,
            utilization: 0.5,
            best_fitness: 1.0,
            items_placed: 5,
            total_items: 10,
            elapsed_ms: 100,
            phase: "Test".to_string(),
            running: true,
        };

        assert!(!wrapper.invoke(&progress));
        assert!(wrapper.is_cancelled());

        // Subsequent calls should also return false
        assert!(!wrapper.invoke(&progress));
    }

    #[test]
    fn test_callback_wrapper_with_user_data() {
        unsafe extern "C" fn counting_callback(
            _json: *const c_char,
            data: *mut c_void,
        ) -> i32 {
            let counter = data as *mut AtomicU32;
            (*counter).fetch_add(1, Ordering::Relaxed);
            1 // Continue
        }

        let counter = Arc::new(AtomicU32::new(0));
        let counter_ptr = Arc::as_ptr(&counter) as *mut c_void;

        let wrapper = unsafe { CallbackWrapper::new(Some(counting_callback), counter_ptr) };

        let progress = ProgressJson {
            iteration: 1,
            total_iterations: 10,
            utilization: 0.5,
            best_fitness: 1.0,
            items_placed: 5,
            total_items: 10,
            elapsed_ms: 100,
            phase: "Test".to_string(),
            running: true,
        };

        wrapper.invoke(&progress);
        wrapper.invoke(&progress);
        wrapper.invoke(&progress);

        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_progress_json_from_progress_info() {
        let info = u_nesting_core::solver::ProgressInfo {
            iteration: 5,
            total_iterations: 50,
            utilization: 0.8,
            best_fitness: 2.0,
            items_placed: 8,
            total_items: 10,
            elapsed_ms: 250,
            phase: "GA Optimization".to_string(),
            running: true,
        };

        let json: ProgressJson = info.into();
        assert_eq!(json.iteration, 5);
        assert_eq!(json.total_iterations, 50);
        assert_eq!(json.utilization, 0.8);
        assert_eq!(json.phase, "GA Optimization");
    }
}
