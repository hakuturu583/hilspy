//! Compiles `profiles/*.txtpb` (protobuf text format) into the binary
//! `ProfileSet` hilspy embeds. Shared by `examples/lidar_profiles.rs` (which
//! writes `profiles.binpb`) and `tests/lidar_profiles.rs` (which checks it is
//! up to date).

use std::path::{Path, PathBuf};

use prost_reflect::{DescriptorPool, DynamicMessage};

pub fn lidar_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("lidar")
}

pub fn binary_path() -> PathBuf {
    lidar_dir().join("profiles.binpb")
}

/// The text profiles, sorted by file name.
pub fn sources() -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(lidar_dir().join("profiles"))
        .expect("lidar/profiles")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "txtpb"))
        .collect();
    files.sort();
    files
}

/// The serialized `ProfileSet` of all text profiles: each file is a
/// `ProfileSet`, and concatenated messages merge their repeated fields.
pub fn compile() -> Result<Vec<u8>, String> {
    let fds = protox::compile(["lidar_profile.proto"], [lidar_dir()]).map_err(|e| e.to_string())?;
    let pool = DescriptorPool::from_file_descriptor_set(fds).map_err(|e| e.to_string())?;
    let set = pool
        .get_message_by_name("hilspy.lidar.ProfileSet")
        .ok_or("no hilspy.lidar.ProfileSet")?;
    let mut out = Vec::new();
    for path in sources() {
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let message = DynamicMessage::parse_text_format(set.clone(), &text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        out.extend(prost::Message::encode_to_vec(&message));
    }
    Ok(out)
}
