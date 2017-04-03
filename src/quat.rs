
/*
use num_traits::identities::Zero;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
use std::default::Default;

use super::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct AxisAngle<F> {
    pub axis: Vec3<F>,
    pub angle: F,
}

 */

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Quat<F> {
    pub x: F,
    pub y: F,
    pub z: F,
    pub w: F,
}

/*
impl<F> From<AxisAngle<F>> for Quat<F> {
    fn from(aa: AxisAngle<F>) -> Quat<F> {
    }
}

impl<F> From<Quat<F>> for AxisAngle<F> {
    fn from(q: Quat<F>) -> AxisAngle<F> {
    }
}
*/
