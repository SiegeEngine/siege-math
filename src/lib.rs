
extern crate num_traits;
#[macro_use]
extern crate serde_derive;
extern crate float_cmp;

pub mod vector;
pub use self::vector::{Vec2, Vec3, Vec4,
                       Direction2, Direction3,
                       X_AXIS_F32, Y_AXIS_F32, Z_AXIS_F32,
                       X_AXIS_F64, Y_AXIS_F64, Z_AXIS_F64,
                       Point2, Point3};

pub mod matrix;
pub use self::matrix::{Mat2, Mat3, Mat4};

pub mod quat;
pub use self::quat::{Quat, NQuat};

pub mod angle;
pub use self::angle::Angle;

pub mod position;
pub use self::position::Position;

use num_traits::{Float, FloatConst, NumAssignOps, NumCast};
use float_cmp::{Ulps, ApproxEq};

// This trait allows us to write code generic across both
// floating point types
pub trait FullFloat: Float + FloatConst + Default +
    NumAssignOps + NumCast + Ulps + ApproxEq<Flt=Self>
{ }
impl FullFloat for f32 {}
impl FullFloat for f64 {}
