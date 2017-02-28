
extern crate num_traits;
#[macro_use]
extern crate serde_derive;
extern crate float_cmp;
// FIXME:  once simd is part of std and stable, use it
// extern crate simd;

pub mod vector;
pub use self::vector::{Vec2, Vec3, Vec4, vec2, vec3, vec4};

pub mod matrix;
pub use self::matrix::{Mat2, Mat3, Mat4};
