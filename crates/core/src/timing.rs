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

/// Whether `limit` (if any) has elapsed since `start`.
pub(crate) fn expired(start: &Timer, limit: Option<web_time::Duration>) -> bool {
    limit.is_some_and(|limit| start.elapsed() > limit)
}

/// Evaluates `items` a batch at a time, stopping once `limit` has elapsed since
/// `start`, and drops the items it did not reach.
///
/// A runner that evaluates a whole population before looking at the clock
/// overruns its time limit by that whole population — many seconds when one
/// evaluation is a full placement. Checking between batches bounds the overrun
/// to one batch. The first batch is always evaluated, so at least one item
/// survives. A batch is as many items as there are worker threads, so parallel
/// evaluation keeps its throughput.
pub(crate) fn evaluate_within<T>(
    items: &mut Vec<T>,
    start: &Timer,
    limit: Option<web_time::Duration>,
    mut evaluate: impl FnMut(&mut [T]),
) {
    #[cfg(feature = "parallel")]
    let batch = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "parallel"))]
    let batch = 1;

    let mut done = 0;
    while done < items.len() {
        let end = (done + batch).min(items.len());
        evaluate(&mut items[done..end]);
        done = end;
        if expired(start, limit) {
            break;
        }
    }
    items.truncate(done);
}

#[cfg(test)]
mod tests {
    use super::*;
    use web_time::Duration;

    #[test]
    fn evaluation_stops_at_the_limit_and_keeps_what_it_evaluated() {
        let start = Timer::now();
        let mut items: Vec<u32> = (0..1000).collect();
        evaluate_within(
            &mut items,
            &start,
            Some(Duration::from_millis(20)),
            |batch| {
                std::thread::sleep(Duration::from_millis(5));
                batch.iter_mut().for_each(|x| *x += 1_000_000);
            },
        );
        assert!(
            !items.is_empty() && items.len() < 1000,
            "evaluated {}",
            items.len()
        );
        assert!(
            items.iter().all(|&x| x >= 1_000_000),
            "an unevaluated item survived"
        );
    }

    #[test]
    fn without_a_limit_everything_is_evaluated() {
        let start = Timer::now();
        let mut items = vec![0u8; 50];
        evaluate_within(&mut items, &start, None, |batch| {
            batch.iter_mut().for_each(|x| *x = 1)
        });
        assert_eq!(items, vec![1u8; 50]);
    }
}
