//! Generates the LiDAR profile types from `lidar/lidar_profile.proto` with
//! protox (a Rust protobuf compiler), so building needs no `protoc`.

fn main() {
    println!("cargo:rerun-if-changed=lidar/lidar_profile.proto");
    let files = protox::compile(["lidar_profile.proto"], ["lidar"])
        .expect("lidar/lidar_profile.proto does not compile");
    prost_build::Config::new()
        .compile_fds(files)
        .expect("cannot generate the LiDAR profile types");
}
