//! Regenerate `lidar/profiles.binpb` from `lidar/profiles/*.txtpb`:
//!
//! ```sh
//! cargo run -p hilspy-core --example lidar_profiles
//! ```

#[path = "../lidar/compile.rs"]
mod compile;

fn main() {
    let bytes = compile::compile().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let path = compile::binary_path();
    std::fs::write(&path, &bytes).expect("cannot write profiles.binpb");
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
}
