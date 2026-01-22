//! Memory optimization utilities.
//!
//! This module provides memory-efficient data structures and allocation patterns
//! for high-performance nesting and packing operations.
//!
//! ## Features
//!
//! - **Object Pool**: Reusable object allocation to reduce heap pressure
//! - **Geometry Instancing**: Shared vertex data for repeated geometries
//! - **Scratch Buffer**: Thread-local temporary storage

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// A simple object pool for reusing allocations.
///
/// This pool helps reduce allocation overhead when repeatedly creating
/// and destroying objects of the same type during optimization loops.
///
/// # Example
///
/// ```rust
/// use u_nesting_core::memory::ObjectPool;
///
/// let pool: ObjectPool<Vec<f64>> = ObjectPool::new(|| Vec::with_capacity(100));
///
/// // Get an object from the pool (or create new if empty)
/// let mut vec = pool.get();
/// vec.push(1.0);
/// vec.push(2.0);
///
/// // Return to pool for reuse (clears the vec)
/// pool.put(vec);
///
/// // Next get() will reuse the allocation
/// let vec2 = pool.get();
/// assert_eq!(vec2.capacity(), 100); // Capacity preserved
/// ```
pub struct ObjectPool<T> {
    pool: RefCell<Vec<T>>,
    factory: Box<dyn Fn() -> T>,
    max_size: usize,
}

impl<T> ObjectPool<T> {
    /// Creates a new object pool with the given factory function.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            pool: RefCell::new(Vec::new()),
            factory: Box::new(factory),
            max_size: 64,
        }
    }

    /// Creates a pool with a custom maximum size.
    pub fn with_max_size<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            pool: RefCell::new(Vec::new()),
            factory: Box::new(factory),
            max_size,
        }
    }

    /// Gets an object from the pool, or creates a new one if the pool is empty.
    pub fn get(&self) -> T {
        self.pool
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| (self.factory)())
    }

    /// Returns an object to the pool for reuse.
    pub fn put(&self, item: T) {
        let mut pool = self.pool.borrow_mut();
        if pool.len() < self.max_size {
            pool.push(item);
        }
        // If pool is full, item is dropped
    }

    /// Returns the current number of objects in the pool.
    pub fn len(&self) -> usize {
        self.pool.borrow().len()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.pool.borrow().is_empty()
    }

    /// Clears all objects from the pool.
    pub fn clear(&self) {
        self.pool.borrow_mut().clear();
    }
}

/// A clearable trait for objects that can be reset for reuse.
pub trait Clearable {
    /// Clears the object's contents while preserving capacity.
    fn clear_for_reuse(&mut self);
}

impl<T> Clearable for Vec<T> {
    fn clear_for_reuse(&mut self) {
        self.clear();
    }
}

impl<K, V> Clearable for HashMap<K, V> {
    fn clear_for_reuse(&mut self) {
        self.clear();
    }
}

/// Object pool that automatically clears returned objects.
pub struct ClearingPool<T: Clearable> {
    inner: ObjectPool<T>,
}

impl<T: Clearable> ClearingPool<T> {
    /// Creates a new clearing pool.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            inner: ObjectPool::new(factory),
        }
    }

    /// Gets an object from the pool.
    pub fn get(&self) -> T {
        self.inner.get()
    }

    /// Returns an object to the pool, clearing it first.
    pub fn put(&self, mut item: T) {
        item.clear_for_reuse();
        self.inner.put(item);
    }
}

/// Shared geometry data for instancing.
///
/// When placing multiple copies of the same geometry, this structure
/// allows sharing the vertex data to reduce memory usage.
#[derive(Debug, Clone)]
pub struct SharedGeometry<V> {
    /// Unique identifier
    pub id: String,
    /// Shared vertex data
    vertices: Arc<Vec<V>>,
    /// Reference count (for debugging/monitoring)
    ref_count: usize,
}

impl<V: Clone> SharedGeometry<V> {
    /// Creates a new shared geometry.
    pub fn new(id: impl Into<String>, vertices: Vec<V>) -> Self {
        Self {
            id: id.into(),
            vertices: Arc::new(vertices),
            ref_count: 1,
        }
    }

    /// Gets a reference to the shared vertices.
    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }

    /// Creates a clone that shares the vertex data.
    pub fn share(&self) -> Self {
        Self {
            id: self.id.clone(),
            vertices: Arc::clone(&self.vertices),
            ref_count: self.ref_count + 1,
        }
    }

    /// Returns the number of references to this geometry's data.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.vertices)
    }
}

/// A geometry instance cache for deduplication.
///
/// Caches shared geometry data to avoid duplicating vertex arrays
/// when the same geometry appears multiple times.
#[derive(Default)]
pub struct GeometryCache<V> {
    cache: HashMap<String, SharedGeometry<V>>,
}

impl<V: Clone> GeometryCache<V> {
    /// Creates a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Gets or creates a shared geometry.
    ///
    /// If a geometry with the same ID exists, returns a shared reference.
    /// Otherwise, creates a new entry with the provided vertices.
    pub fn get_or_insert(&mut self, id: &str, vertices: Vec<V>) -> SharedGeometry<V> {
        if let Some(existing) = self.cache.get(id) {
            existing.share()
        } else {
            let shared = SharedGeometry::new(id, vertices);
            self.cache.insert(id.to_string(), shared.share());
            shared
        }
    }

    /// Checks if a geometry is already cached.
    pub fn contains(&self, id: &str) -> bool {
        self.cache.contains_key(id)
    }

    /// Returns the number of cached geometries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Returns total memory usage estimate (vertex count * size).
    pub fn memory_usage(&self) -> usize {
        self.cache
            .values()
            .map(|g| g.vertices.len() * std::mem::size_of::<V>())
            .sum()
    }
}

/// Thread-local scratch buffer for temporary allocations.
///
/// Provides a reusable buffer for algorithms that need temporary storage,
/// avoiding repeated allocations.
///
/// # Example
///
/// ```rust
/// use u_nesting_core::memory::ScratchBuffer;
///
/// let scratch = ScratchBuffer::<f64>::new(1024);
///
/// // Use the buffer
/// scratch.with_buffer(|buf| {
///     buf.push(1.0);
///     buf.push(2.0);
///     // Process data...
///     buf.iter().sum::<f64>()
/// });
/// // Buffer is automatically cleared after use
/// ```
pub struct ScratchBuffer<T> {
    buffer: RefCell<Vec<T>>,
    initial_capacity: usize,
}

impl<T> ScratchBuffer<T> {
    /// Creates a new scratch buffer with the given initial capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RefCell::new(Vec::with_capacity(capacity)),
            initial_capacity: capacity,
        }
    }

    /// Executes a function with access to the buffer, clearing it afterward.
    pub fn with_buffer<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Vec<T>) -> R,
    {
        let mut buf = self.buffer.borrow_mut();
        let result = f(&mut buf);
        buf.clear();
        result
    }

    /// Returns the current capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.buffer.borrow().capacity()
    }

    /// Shrinks the buffer back to initial capacity if it has grown too large.
    pub fn shrink_if_needed(&self, max_capacity: usize) {
        let mut buf = self.buffer.borrow_mut();
        if buf.capacity() > max_capacity {
            buf.shrink_to(self.initial_capacity);
        }
    }
}

/// Memory statistics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Number of pool allocations avoided
    pub pool_hits: usize,
    /// Number of new allocations made
    pub pool_misses: usize,
    /// Number of cached geometries
    pub cached_geometries: usize,
    /// Estimated memory saved by caching (bytes)
    pub memory_saved_bytes: usize,
}

impl MemoryStats {
    /// Creates new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a pool hit (reused allocation).
    pub fn record_pool_hit(&mut self) {
        self.pool_hits += 1;
    }

    /// Records a pool miss (new allocation).
    pub fn record_pool_miss(&mut self) {
        self.pool_misses += 1;
    }

    /// Returns the pool hit rate (0.0 - 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.pool_hits + self.pool_misses;
        if total == 0 {
            0.0
        } else {
            self.pool_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let pool: ObjectPool<Vec<i32>> = ObjectPool::new(|| Vec::with_capacity(10));

        assert!(pool.is_empty());

        let mut v1 = pool.get();
        v1.push(1);
        v1.push(2);
        pool.put(v1);

        assert_eq!(pool.len(), 1);

        let v2 = pool.get();
        assert!(pool.is_empty());
        assert!(v2.capacity() >= 10);
    }

    #[test]
    fn test_clearing_pool() {
        let pool: ClearingPool<Vec<i32>> = ClearingPool::new(|| Vec::with_capacity(10));

        let mut v1 = pool.get();
        v1.push(1);
        v1.push(2);
        v1.push(3);
        pool.put(v1);

        let v2 = pool.get();
        assert!(v2.is_empty()); // Should be cleared
        assert!(v2.capacity() >= 10); // But capacity preserved
    }

    #[test]
    fn test_shared_geometry() {
        let g1 = SharedGeometry::new("test", vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
        assert_eq!(g1.strong_count(), 1);

        let g2 = g1.share();
        assert_eq!(g1.strong_count(), 2);
        assert_eq!(g2.strong_count(), 2);

        assert_eq!(g1.vertices().len(), 3);
        assert_eq!(g2.vertices().len(), 3);
    }

    #[test]
    fn test_geometry_cache() {
        let mut cache: GeometryCache<(f64, f64)> = GeometryCache::new();

        let g1 = cache.get_or_insert(
            "rect1",
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let g2 = cache.get_or_insert("rect1", vec![]); // Should reuse existing

        assert_eq!(g1.strong_count(), 3); // cache + g1 + g2
        assert_eq!(g2.strong_count(), 3);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_scratch_buffer() {
        let scratch = ScratchBuffer::<i32>::new(100);

        let sum = scratch.with_buffer(|buf| {
            buf.push(1);
            buf.push(2);
            buf.push(3);
            buf.iter().sum::<i32>()
        });

        assert_eq!(sum, 6);

        // Buffer should be cleared
        scratch.with_buffer(|buf| {
            assert!(buf.is_empty());
        });
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();

        stats.record_pool_hit();
        stats.record_pool_hit();
        stats.record_pool_miss();

        assert_eq!(stats.pool_hits, 2);
        assert_eq!(stats.pool_misses, 1);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }
}
