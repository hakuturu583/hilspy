//! The embedded binary profiles match their text sources.

#[path = "../lidar/compile.rs"]
mod compile;

#[test]
fn binary_profiles_are_up_to_date() {
    let compiled = compile::compile().unwrap();
    let committed = std::fs::read(compile::binary_path()).unwrap();
    assert!(
        compiled == committed,
        "lidar/profiles.binpb is out of date: \
         run `cargo run -p hilspy-core --example lidar_profiles`"
    );
}

#[test]
fn every_default_profile_is_valid() {
    let names = hilspy_core::lidar::names();
    assert!(names.len() >= 17, "{names:?}");
    for name in names {
        hilspy_core::lidar::Model::find(&name, None).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}
