

pub mod point;
pub use self::point::{Point2, Point3};

pub mod direction;
pub use self::direction::{Direction2, Direction3,
                          X_AXIS_F32, Y_AXIS_F32, Z_AXIS_F32,
                          X_AXIS_F64, Y_AXIS_F64, Z_AXIS_F64};

use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
use std::default::Default;
use num_traits::NumCast;
use float_cmp::{Ulps, ApproxEq};
use FullFloat;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vec2<F> {
    pub x: F,
    pub y: F,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vec3<F> {
    pub x: F,
    pub y: F,
    pub z: F,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vec4<F> {
    pub x: F,
    pub y: F,
    pub z: F,
    pub w: F,
}

// -- indexing ----------------------------------------------------------------

impl<F: FullFloat> Index<usize> for Vec2<F> {
    type Output = F;

    #[inline]
    fn index(&self, i: usize) -> &F {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl<F: FullFloat> Index<usize> for Vec3<F> {
    type Output = F;

    #[inline]
    fn index(&self, i: usize) -> &F {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<F: FullFloat> Index<usize> for Vec4<F> {
    type Output = F;

    #[inline]
    fn index(&self, i: usize) -> &F {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl<F: FullFloat> IndexMut<usize> for Vec2<F> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut F {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl<F: FullFloat> IndexMut<usize> for Vec3<F> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut F {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<F: FullFloat> IndexMut<usize> for Vec4<F> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut F {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

// -- dropping a dimension ----------------------------------------------------

impl<F: FullFloat> Vec3<F> {
    #[inline]
    pub fn truncate_n(&self, n: usize) -> Vec2<F> {
        match n {
            0 => Vec2::new(self.y, self.z),
            1 => Vec2::new(self.x, self.z),
            2 => Vec2::new(self.x, self.y),
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<F: FullFloat> Vec3<F> {
    #[inline]
    pub fn truncate_x(&self) -> Vec2<F> {
        Vec2::new(self.y, self.z)
    }
    #[inline]
    pub fn truncate_y(&self) -> Vec2<F> {
        Vec2::new(self.x, self.z)
    }
    #[inline]
    pub fn truncate_z(&self) -> Vec2<F> {
        Vec2::new(self.x, self.y)
    }
}

impl<F: FullFloat> Vec4<F> {
    #[inline]
    pub fn truncate_n(&self, n: usize) -> Vec3<F> {
        match n {
            0 => Vec3::new(self.y, self.z, self.w),
            1 => Vec3::new(self.x, self.z, self.w),
            2 => Vec3::new(self.x, self.y, self.w),
            3 => Vec3::new(self.x, self.y, self.z),
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl<F: FullFloat> Vec4<F> {
    #[inline]
    pub fn truncate_x(&self) -> Vec3<F> {
        Vec3::new(self.y, self.z, self.w)
    }
    #[inline]
    pub fn truncate_y(&self) -> Vec3<F> {
        Vec3::new(self.x, self.z, self.w)
    }
    #[inline]
    pub fn truncate_z(&self) -> Vec3<F> {
        Vec3::new(self.x, self.y, self.w)
    }
    #[inline]
    pub fn truncate_w(&self) -> Vec3<F> {
        Vec3::new(self.x, self.y, self.z)
    }
}

// ----------------------------------------------------------------------------

macro_rules! impl_vector {
    ($VecN:ident { $first:ident, $($field:ident),* }) => {
        impl<F: FullFloat> $VecN<F> {
            /// Construct a new vector
            #[inline]
            pub fn new($first: F, $($field: F),*) -> $VecN<F> {
                $VecN { $first: $first, $($field: $field),* }
            }
        }

        impl<F: FullFloat> $VecN<F> {
            #[inline]
            pub fn zero() -> $VecN<F> {
                $VecN { $first: F::zero(), $($field: F::zero()),* }
            }
        }

        impl<F: FullFloat> Default for $VecN<F> {
            #[inline]
            fn default() -> $VecN<F> {
                $VecN { $first: F::default(), $($field: F::default()),* }
            }
        }

        impl<F: FullFloat> $VecN<F>{
            #[inline]
            pub fn squared_magnitude(&self) -> F {
                self.$first * self.$first $(+ self.$field * self.$field)*
            }
        }

        impl<F: FullFloat> $VecN<F> {
            #[inline]
            pub fn magnitude(&self) -> F {
                self.squared_magnitude().sqrt()
                // FIXME: once simd is part of std and stable, use it
                // rsqrt is faster than sqrt (but is approximate)
                // self.squared_magnitude().rsqrt()
            }
        }

        impl<F: FullFloat> $VecN<F> {
            pub fn is_normal(&self) -> bool {
                self.magnitude().approx_eq(
                    &F::one(),
                    NumCast::from(10_u32).unwrap(),
                    <F as NumCast>::from(10.0_f32).unwrap() * F::epsilon()
                )
            }
        }

        impl<F: FullFloat> Mul<F> for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn mul(self, rhs: F) -> $VecN<F> {
                $VecN {
                    $first: self.$first * rhs,
                    $($field: self.$field * rhs),*
                }
            }
        }

        impl<F: FullFloat> MulAssign<F> for $VecN<F> {
            #[inline]
            fn mul_assign(&mut self, rhs: F) {
                self.$first *= rhs;
                $(self.$field *= rhs);*
            }
        }

        impl<F: FullFloat> Div<F> for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn div(self, rhs: F) -> $VecN<F> {
                $VecN {
                    $first: self.$first / rhs,
                    $($field: self.$field / rhs),*
                }
            }
        }

        impl<F: FullFloat> DivAssign<F> for $VecN<F> {
            #[inline]
            fn div_assign(&mut self, rhs: F) {
                self.$first /= rhs;
                $(self.$field /= rhs);*
            }
        }

        impl<F: FullFloat> Neg for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn neg(self) -> $VecN<F> {
                $VecN {
                    $first: -self.$first,
                    $($field: -self.$field),*
                }
            }
        }

        impl<F: FullFloat> Add for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn add(self, other: $VecN<F>) -> $VecN<F> {
                $VecN {
                    $first: self.$first + other.$first,
                    $($field: self.$field + other.$field),*
                }
            }
        }

        impl<F: FullFloat> AddAssign<$VecN<F>> for $VecN<F> {
            #[inline]
            fn add_assign(&mut self, other: $VecN<F>) {
                self.$first += other.$first;
                $(self.$field += other.$field);*
            }
        }

        impl<F: FullFloat> Sub for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn sub(self, other: $VecN<F>) -> $VecN<F> {
                $VecN {
                    $first: self.$first - other.$first,
                    $($field: self.$field - other.$field),*
                }
            }
        }

        impl<F: FullFloat> SubAssign<$VecN<F>> for $VecN<F> {
            #[inline]
            fn sub_assign(&mut self, other: $VecN<F>) {
                self.$first -= other.$first;
                $(self.$field -= other.$field);*
            }
        }

        impl<F: FullFloat> $VecN<F> {
            #[inline]
            pub fn dot(&self, rhs: $VecN<F>) -> F {
                self.$first * rhs.$first
                    $(+ self.$field * rhs.$field)*
            }
        }

        impl<F: FullFloat> $VecN<F> {
            #[inline]
            pub fn project_onto(&self, axis: $VecN<F>) -> $VecN<F> {
                axis * (self.dot(axis) / axis.dot(axis))
            }
        }

        impl<F: FullFloat> $VecN<F>
        {
            #[inline]
            pub fn reject_onto(&self, axis: $VecN<F>) -> $VecN<F> {
                *self - self.project_onto(axis)
            }
        }
    }
}

impl_vector!(Vec2 { x, y });
impl_vector!(Vec3 { x, y, z });
impl_vector!(Vec4 { x, y, z, w });

// ----------------------------------------------------------------------------

impl<F: FullFloat> Vec3<F> {
    #[inline]
    pub fn cross(&self, rhs: Vec3<F>) -> Vec3<F> {
        Vec3::new(
            self.y*rhs.z - self.z*rhs.y,
            self.z*rhs.x - self.x*rhs.z,
            self.x*rhs.y - self.y*rhs.x
        )
    }
}

impl<F: FullFloat> Vec3<F> {
    #[inline]
    pub fn triple_product(&self, b: Vec3<F>, c: Vec3<F>) -> F {
        self.cross(b).dot(c)
    }
}

// ----------------------------------------------------------------------------
// Shortening

impl<F: FullFloat> From<Vec4<F>> for Vec3<F> {
    fn from(v: Vec4<F>) -> Vec3<F> {
        Vec3 { x: v.x, y: v.y, z: v.z }
    }
}

impl<F: FullFloat> From<Vec3<F>> for Vec2<F> {
    fn from(v: Vec3<F>) -> Vec2<F> {
        Vec2 { x: v.x, y: v.y }
    }
}

// ----------------------------------------------------------------------------
// casting between float types

/*
Unfortunately I cant get these to work because F and G could be the same type
and that collides with an impl in the standard library.  I have not figured
out how to tell rust that F and G are not the same type.

impl<F: FullFloat, G: FullFloat> From<Vec2<F>> for Vec2<G>
{
    fn from(v: Vec2<F>) -> Vec2<G> {
        Vec2 { x: v.x.as_(), y: v.y.as_() }
    }
}

impl<F: FullFloat, G: FullFloat> From<Vec3<F>> for Vec3<G> {
    fn from(v: Vec3<F>) -> Vec3<G> {
        Vec3 { x: v.x.as_(), y: v.y.as_(), z: v.z.as_() }
    }
}

impl<F: FullFloat, G: FullFloat> From<Vec4<F>> for Vec4<G> {
    fn from(v: Vec4<F>) -> Vec4<G> {
        Vec4 { x: v.x.as_(), y: v.y.as_(), z: v.z.as_(), w: v.w.as_() }
    }
}
*/

impl From<Vec2<f64>> for Vec2<f32> {
    fn from(v: Vec2<f64>) -> Vec2<f32> {
        Vec2 { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vec2<f32>> for Vec2<f64> {
    fn from(v: Vec2<f32>) -> Vec2<f64> {
        Vec2 { x: v.x as f64, y: v.y as f64 }
    }
}

impl From<Vec3<f64>> for Vec3<f32> {
    fn from(v: Vec3<f64>) -> Vec3<f32> {
        Vec3 { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vec3<f32>> for Vec3<f64> {
    fn from(v: Vec3<f32>) -> Vec3<f64> {
        Vec3 { x: v.x as f64, y: v.y as f64, z: v.z as f64 }
    }
}

impl From<Vec4<f64>> for Vec4<f32> {
    fn from(v: Vec4<f64>) -> Vec4<f32> {
        Vec4 { x: v.x as f32, y: v.y as f32, z: v.z as f32, w: v.w as f32 }
    }
}

impl From<Vec4<f32>> for Vec4<f64> {
    fn from(v: Vec4<f32>) -> Vec4<f64> {
        Vec4 { x: v.x as f64, y: v.y as f64, z: v.z as f64, w: v.w as f64 }
    }
}

// ----------------------------------------------------------------------------
// Approx Eq

impl<F: FullFloat> ApproxEq for Vec2<F> {
    type Flt = F;

    fn approx_eq(&self, other: &Self, ulps: <<F as ApproxEq>::Flt as Ulps>::U,
                 epsilon: <F as ApproxEq>::Flt) -> bool
    {
        self.x.approx_eq(&other.x, ulps, epsilon)
            && self.y.approx_eq(&other.y, ulps, epsilon)
    }
}

impl<F: FullFloat> ApproxEq for Vec3<F> {
    type Flt = F;

    fn approx_eq(&self, other: &Self, ulps: <<F as ApproxEq>::Flt as Ulps>::U,
                 epsilon: <F as ApproxEq>::Flt) -> bool
    {
        self.x.approx_eq(&other.x, ulps, epsilon)
            && self.y.approx_eq(&other.y, ulps, epsilon)
            && self.z.approx_eq(&other.z, ulps, epsilon)
    }
}

impl<F: FullFloat> ApproxEq for Vec4<F> {
    type Flt = F;

    fn approx_eq(&self, other: &Self, ulps: <<F as ApproxEq>::Flt as Ulps>::U,
                 epsilon: <F as ApproxEq>::Flt) -> bool
    {
        self.x.approx_eq(&other.x, ulps, epsilon)
            && self.y.approx_eq(&other.y, ulps, epsilon)
            && self.z.approx_eq(&other.z, ulps, epsilon)
            && self.w.approx_eq(&other.w, ulps, epsilon)
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use float_cmp:: ApproxEq;
    use super::Vec2;
    const VEC2: Vec2<f32> = Vec2 { x: 1.0, y: 2.0 };

    #[test]
    fn test_new() {
        assert_eq!(Vec2::new(1.0_f32, 2.0_f32), VEC2);
    }

    #[test]
    fn test_zero() {
        assert_eq!(Vec2::new(0.0_f32, 0.0_f32), Vec2::zero());
        let z: Vec2<f32> = Vec2::zero();
        assert_eq!(z[0], 0.0_f32);
        assert_eq!(z[1], 0.0_f32);
    }

    #[test]
    fn test_squared_magnitude() {
        assert!(VEC2.squared_magnitude().approx_eq(&5.0, 1, 1.0 * ::std::f32::EPSILON));
    }

    #[test]
    fn test_index() {
        assert_eq!(VEC2[0], VEC2.x);
        assert_eq!(VEC2[1], VEC2.y);
    }

    #[test]
    fn test_index_mut() {
        let mut v: Vec2<f32> = Vec2::new(3.0, 5.0);
        v[1] = 6.0;
        assert_eq!(v.y, 6.0);
    }
}
