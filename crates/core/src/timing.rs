//! WASM-compatible timing abstraction.
//!
//! Wall-clock timing backed by [`web_time::Instant`], which transparently maps
//! to [`std::time::Instant`] on native targets and to `performance.now()` on
//! `wasm32`. This keeps time-based termination (`time_limit_ms`) working on both
//! native and WASM — iteration limits remain the safety net that bounds every
//! algorithm regardless of clock resolution.
//!
//! A previous WASM build used a no-op timer that always reported zero elapsed
//! time; that silently disabled `time_limit_ms` on WASM, so strategies ran to
//! their full iteration cap (multi-second freezes in the browser).

mod inner {
    use web_time::{Duration, Instant};

    /// A wall-clock timer backed by [`web_time::Instant`].
    ///
    /// On native targets this is exactly [`std::time::Instant`]; on `wasm32` it
    /// uses the JS `performance.now()` clock via `web-time`.
    #[derive(Debug, Clone, Copy)]
    pub struct Timer(Instant);

    impl Timer {
        /// Starts the timer.
        pub fn now() -> Self {
            Timer(Instant::now())
        }

        /// Returns the elapsed time since the timer was started.
        pub fn elapsed(&self) -> Duration {
            self.0.elapsed()
        }

        /// Returns elapsed time in milliseconds.
        pub fn elapsed_ms(&self) -> u64 {
            self.0.elapsed().as_millis() as u64
        }
    }
}

pub use inner::Timer;
