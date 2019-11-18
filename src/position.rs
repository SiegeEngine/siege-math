
use serde::{Serialize, Deserialize};
use {Point3, NQuat};

/// A position is a combination of a point and an orientation
///
/// Orientation is more than just a facing vector, it must also resolve
/// which way is up, but does not need yet another vector for that. It
/// turns out a (normalized) Quaterion fits the role perfectly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Position<F> {
    pub point: Point3<F>,
    pub ori: NQuat<F>,
}
