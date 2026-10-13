//! Dataset download and management utilities.
//!
//! Provides functionality to download ESICUP benchmark datasets from GitHub
//! and manage local dataset storage.

use crate::dataset::Dataset;
use crate::parser::DatasetParser;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during dataset download operations.
#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("Failed to parse dataset: {0}")]
    ParseError(#[from] crate::parser::ParseError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Dataset not found: {0}")]
    NotFound(String),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Information about an available ESICUP dataset.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Dataset name (e.g., "SHAPES", "SHIRTS")
    pub name: &'static str,
    /// Available instances within this dataset
    pub instances: &'static [&'static str],
    /// Brief description
    pub description: &'static str,
    /// Best-known packing density (placed area / used strip area, 0..=1) per
    /// instance, from the literature -- not a strip length. A length depends on
    /// the instance's own strip width and piece areas, so it is derived from the
    /// data when the instance is loaded ([`Dataset::length_at_density`]); a
    /// density cannot be out of range the way a hand-copied length was (the
    /// lengths this table held put MARQUES at a density of 105 %).
    ///
    /// Source: Gardeyn & Wauters, *sparrow* (2025), arXiv:2509.13329, Table 3
    /// (best density sparrow produced, at least the previous best for every
    /// instance). BLAZ and BALDACCI are not in that table and carry none.
    pub best_known_density: &'static [(&'static str, f64)],
}

/// All available ESICUP 2D irregular datasets.
pub const ESICUP_DATASETS: &[DatasetInfo] = &[
    DatasetInfo {
        name: "ALBANO",
        instances: &["albano"],
        description: "24 items, convex and non-convex mixed",
        best_known_density: &[("albano", 0.8982)],
    },
    DatasetInfo {
        name: "BLAZ",
        instances: &["blaz1", "blaz2", "blaz3"],
        description: "Simple polygons, 7-28 items",
        best_known_density: &[],
    },
    DatasetInfo {
        name: "DAGLI",
        instances: &["dagli"],
        description: "30 items, industrial patterns",
        best_known_density: &[("dagli", 0.9017)],
    },
    DatasetInfo {
        name: "FU",
        instances: &["fu"],
        description: "12 items, complex shapes",
        best_known_density: &[("fu", 0.9241)],
    },
    DatasetInfo {
        name: "JAKOBS",
        instances: &["jakobs1", "jakobs2"],
        description: "Classic benchmark, 25 items",
        best_known_density: &[("jakobs1", 0.8926), ("jakobs2", 0.8773)],
    },
    DatasetInfo {
        name: "MAO",
        instances: &["mao"],
        description: "20 items, varied sizes",
        best_known_density: &[("mao", 0.8687)],
    },
    DatasetInfo {
        name: "MARQUES",
        instances: &["marques"],
        description: "24 items, some with holes",
        best_known_density: &[("marques", 0.9202)],
    },
    DatasetInfo {
        name: "SHAPES",
        instances: &["shapes0", "shapes1"],
        description: "43 items total, varied geometry",
        best_known_density: &[("shapes0", 0.6998), ("shapes1", 0.7673)],
    },
    DatasetInfo {
        name: "SHIRTS",
        instances: &["shirts"],
        description: "99 items, large industrial dataset",
        best_known_density: &[("shirts", 0.9092)],
    },
    DatasetInfo {
        name: "SWIM",
        instances: &["swim"],
        description: "48 items, curved approximations",
        best_known_density: &[("swim", 0.7983)],
    },
    DatasetInfo {
        name: "TROUSERS",
        instances: &["trousers"],
        description: "64 items, industrial clothing patterns",
        best_known_density: &[("trousers", 0.9262)],
    },
    DatasetInfo {
        name: "BALDACCI",
        instances: &["baldacci"],
        description: "Alternative industrial dataset",
        best_known_density: &[],
    },
];

/// Manager for downloading and caching ESICUP datasets.
#[derive(Debug)]
pub struct DatasetManager {
    /// Base directory for storing datasets
    base_dir: PathBuf,
    /// Parser instance
    parser: DatasetParser,
}

impl DatasetManager {
    /// Creates a new dataset manager with the given base directory.
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
            parser: DatasetParser::new(),
        }
    }

    /// Creates a dataset manager using the default datasets directory.
    pub fn default_location() -> Self {
        Self::new("datasets/2d/esicup")
    }

    /// Returns the base directory path.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Ensures the base directory exists.
    pub fn ensure_dir(&self) -> Result<(), DownloadError> {
        fs::create_dir_all(&self.base_dir)?;
        Ok(())
    }

    /// Lists all available datasets.
    pub fn list_available() -> &'static [DatasetInfo] {
        ESICUP_DATASETS
    }

    /// Gets information about a specific dataset.
    pub fn get_dataset_info(name: &str) -> Option<&'static DatasetInfo> {
        ESICUP_DATASETS
            .iter()
            .find(|d| d.name.eq_ignore_ascii_case(name))
    }

    /// Checks if a dataset instance is cached locally.
    pub fn is_cached(&self, instance: &str) -> bool {
        self.get_cache_path(instance).exists()
    }

    /// Gets the cache file path for an instance.
    pub fn get_cache_path(&self, instance: &str) -> PathBuf {
        self.base_dir
            .join(format!("{}.json", instance.to_lowercase()))
    }

    /// Downloads a dataset instance from GitHub.
    ///
    /// If the dataset is already cached, returns the cached version.
    pub fn download(&self, dataset: &str, instance: &str) -> Result<Dataset, DownloadError> {
        self.ensure_dir()?;

        let cache_path = self.get_cache_path(instance);

        // Check cache first
        if cache_path.exists() {
            let content = fs::read_to_string(&cache_path)?;
            return Ok(Self::with_reference(
                self.parser.parse_json(&content)?,
                instance,
            ));
        }

        // Download from GitHub
        let ds = self.parser.download_and_parse(dataset, instance)?;

        // Cache locally
        let json = serde_json::to_string_pretty(&ds)?;
        fs::write(&cache_path, json)?;

        Ok(Self::with_reference(ds, instance))
    }

    /// Downloads a dataset instance, optionally forcing a re-download.
    pub fn download_force(
        &self,
        dataset: &str,
        instance: &str,
        force: bool,
    ) -> Result<Dataset, DownloadError> {
        if force {
            let cache_path = self.get_cache_path(instance);
            if cache_path.exists() {
                fs::remove_file(&cache_path)?;
            }
        }
        self.download(dataset, instance)
    }

    /// Loads a dataset from the local cache.
    pub fn load_cached(&self, instance: &str) -> Result<Dataset, DownloadError> {
        let cache_path = self.get_cache_path(instance);
        if !cache_path.exists() {
            return Err(DownloadError::NotFound(instance.to_string()));
        }
        let content = fs::read_to_string(&cache_path)?;
        Ok(Self::with_reference(
            self.parser.parse_json(&content)?,
            instance,
        ))
    }

    /// Sets the dataset's best-known strip length from the instance's
    /// best-known density, so a run's `gap_percent` measures against the
    /// literature. It used to be left unset: no ESICUP run ever had a gap.
    fn with_reference(mut ds: Dataset, instance: &str) -> Dataset {
        if ds.best_known.is_none() {
            ds.best_known =
                Self::best_known_density(instance).and_then(|d| ds.length_at_density(d));
        }
        ds
    }

    /// Downloads all instances of a dataset.
    pub fn download_all_instances(&self, dataset: &str) -> Result<Vec<Dataset>, DownloadError> {
        let info = Self::get_dataset_info(dataset)
            .ok_or_else(|| DownloadError::NotFound(dataset.to_string()))?;

        let mut datasets = Vec::new();
        for instance in info.instances {
            datasets.push(self.download(dataset, instance)?);
        }
        Ok(datasets)
    }

    /// Downloads all available ESICUP datasets.
    ///
    /// Returns a list of (dataset_name, instance_name, result) tuples.
    pub fn download_all(&self) -> Vec<(String, String, Result<Dataset, DownloadError>)> {
        let mut results = Vec::new();

        for info in ESICUP_DATASETS {
            for instance in info.instances {
                let result = self.download(info.name, instance);
                results.push((info.name.to_string(), instance.to_string(), result));
            }
        }

        results
    }

    /// Lists all cached dataset instances.
    pub fn list_cached(&self) -> Result<Vec<String>, DownloadError> {
        let mut cached = Vec::new();

        if self.base_dir.exists() {
            for entry in fs::read_dir(&self.base_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    if let Some(stem) = path.file_stem() {
                        cached.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }

        cached.sort();
        Ok(cached)
    }

    /// Clears all cached datasets.
    pub fn clear_cache(&self) -> Result<(), DownloadError> {
        if self.base_dir.exists() {
            for entry in fs::read_dir(&self.base_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }

    /// The best-known packing density of an instance (0..=1), if the
    /// literature table has one.
    pub fn best_known_density(instance: &str) -> Option<f64> {
        for info in ESICUP_DATASETS {
            for (inst, best) in info.best_known_density {
                if inst.eq_ignore_ascii_case(instance) {
                    return Some(*best);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_available() {
        let datasets = DatasetManager::list_available();
        assert!(datasets.len() >= 11);

        let shapes = DatasetManager::get_dataset_info("SHAPES");
        assert!(shapes.is_some());
        let shapes = shapes.unwrap();
        assert_eq!(shapes.instances.len(), 2);
    }

    #[test]
    fn test_best_known() {
        assert_eq!(DatasetManager::best_known_density("shapes0"), Some(0.6998));
        assert_eq!(DatasetManager::best_known_density("jakobs1"), Some(0.8926));
        assert_eq!(DatasetManager::best_known_density("blaz1"), None);
        assert_eq!(DatasetManager::best_known_density("unknown"), None);
        // A density is a fraction of the strip: every one in the table is.
        for info in ESICUP_DATASETS {
            for &(inst, d) in info.best_known_density {
                assert!(d > 0.0 && d <= 1.0, "{inst}: {d}");
                assert!(
                    info.instances.contains(&inst),
                    "{inst} is not an instance of {}",
                    info.name
                );
            }
        }
    }
}
