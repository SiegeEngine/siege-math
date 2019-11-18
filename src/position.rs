
use serde::{Serialize, Deserialize};
use float_cmp::ApproxEq;
use crate::{Point3, NQuat};

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

// ----------------------------------------------------------------------------
// ApproxEq

impl<'a, M: Copy + Default, F: Copy + ApproxEq<Margin=M>> ApproxEq for &'a Position<F> {
    type Margin = M;

    fn approx_eq<T: Into<Self::Margin>>(self, other: Self, margin: T) -> bool {
        let margin = margin.into();
        self.point.approx_eq(&other.point, margin)
            && self.ori.approx_eq(&other.ori, margin)
    }
}
