//! Build script for generating C headers.

fn main() {
    // Generate C header using cbindgen
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

    if let Ok(bindings) = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        // Always write to OUT_DIR (cargo publish compatible)
        let out_path = std::path::Path::new(&out_dir).join("u_nesting.h");
        bindings.write_to_file(&out_path);
    }
}
