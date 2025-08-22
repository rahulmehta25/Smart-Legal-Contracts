//! Build script for Rust core

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to invalidate the built crate whenever the build script changes
    println!("cargo:rerun-if-changed=build.rs");
    
    // Set up environment variables for compilation
    println!("cargo:rustc-env=CARGO_PKG_BUILD_TIMESTAMP={}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    
    // Enable specific CPU features for SIMD if available
    if cfg!(target_arch = "x86_64") {
        println!("cargo:rustc-cfg=target_feature=\"sse2\"");
        println!("cargo:rustc-cfg=target_feature=\"sse4.1\"");
        if is_x86_feature_detected!("avx2") {
            println!("cargo:rustc-cfg=target_feature=\"avx2\"");
        }
    }
    
    // Configure for WebAssembly target
    if cfg!(target_arch = "wasm32") {
        println!("cargo:rustc-cfg=web_sys_unstable_apis");
    }
    
    // Set up Python module name for PyO3
    if cfg!(feature = "python") {
        println!("cargo:rustc-env=PYO3_PYTHON_MODULE_NAME=rust_core");
    }
    
    // Optimization flags for release builds
    if env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-env=RUST_CORE_OPTIMIZED=1");
        
        // Link-time optimization
        println!("cargo:rustc-link-arg=-fuse-ld=lld");
        println!("cargo:rustc-link-arg=-Wl,--icf=all");
    }
    
    // Generate build information
    let out_dir = env::var("OUT_DIR").unwrap();
    let build_info_path = PathBuf::from(out_dir).join("build_info.rs");
    
    let build_info = format!(
        r#"
        pub const BUILD_TIMESTAMP: &str = "{}";
        pub const GIT_HASH: &str = "{}";
        pub const RUST_VERSION: &str = "{}";
        pub const TARGET_TRIPLE: &str = "{}";
        pub const OPTIMIZATION_LEVEL: &str = "{}";
        "#,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        env::var("VERGEN_GIT_SHA").unwrap_or_else(|_| "unknown".to_string()),
        env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
        env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_string()),
    );
    
    std::fs::write(&build_info_path, build_info).expect("Failed to write build info");
    
    println!("cargo:rerun-if-env-changed=RUSTC_VERSION");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=OPT_LEVEL");
}

// Helper function for CPU feature detection (would need actual implementation)
fn is_x86_feature_detected!(_feature: &str) -> bool {
    // Placeholder - would use actual CPU feature detection
    false
}