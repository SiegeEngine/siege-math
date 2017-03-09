
use num_traits::identities::Zero;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
use std::default::Default;

use super::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct AxisAngle<T> {
    pub axis: Vec3<T>,
    pub angle: T,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Quat<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

/*
impl<T> From<AxisAngle<T>> for Quat<T> {
    fn from(aa: AxisAngle<T>) -> Quat<T> {
    }
}

impl<T> From<Quat<T>> for AxisAngle<T> {
    fn from(q: Quat<T>) -> AxisAngle<T> {
    }
}
*/
