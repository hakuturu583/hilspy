use super::frame::Frame;

/// A receive filter: a frame passes when `frame.id & mask == id & mask`
/// (and, when `extended` is set, the identifier kind matches). A bus with no
/// filters receives every frame; with several, a frame passes if any matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filter {
    pub id: u32,
    pub mask: u32,
    /// `Some(true)`: 29-bit identifiers only; `Some(false)`: 11-bit only;
    /// `None`: either.
    pub extended: Option<bool>,
}

impl Filter {
    pub fn matches(&self, frame: &Frame) -> bool {
        if let Some(extended) = self.extended {
            if extended != frame.extended {
                return false;
            }
        }
        frame.id & self.mask == self.id & self.mask
    }

    pub(crate) fn accepts(filters: &[Filter], frame: &Frame) -> bool {
        filters.is_empty() || filters.iter().any(|f| f.matches(frame))
    }
}
