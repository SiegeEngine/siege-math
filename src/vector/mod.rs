
pub mod point;
pub use self::point::{Point2, Point3};

pub mod direction;
pub use self::direction::{Direction2, Direction3};

use num_traits::identities::Zero;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
use std::default::Default;

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

impl<F> Index<usize> for Vec2<F> {
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

impl<F> Index<usize> for Vec3<F> {
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

impl<F> Index<usize> for Vec4<F> {
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

impl<F> IndexMut<usize> for Vec2<F> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut F {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl<F> IndexMut<usize> for Vec3<F> {
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

impl<F> IndexMut<usize> for Vec4<F> {
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

impl<F: Copy> Vec3<F> {
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

impl<F: Copy> Vec3<F> {
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

impl<F: Copy> Vec4<F> {
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

impl<F: Copy> Vec4<F> {
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
        impl<F> $VecN<F> {
            /// Construct a new vector
            #[inline]
            pub fn new($first: F, $($field: F),*) -> $VecN<F> {
                $VecN { $first: $first, $($field: $field),* }
            }
        }

        impl<F: Zero> $VecN<F> {
            #[inline]
            pub fn zero() -> $VecN<F> {
                $VecN { $first: F::zero(), $($field: F::zero()),* }
            }
        }

        impl<F: Default> Default for $VecN<F> {
            #[inline]
            fn default() -> $VecN<F> {
                $VecN { $first: F::default(), $($field: F::default()),* }
            }
        }

        impl<F: Copy + Mul<F,Output=F> + Add<F,Output=F>> $VecN<F>{
            #[inline]
            pub fn squared_magnitude(&self) -> F {
                self.$first * self.$first $(+ self.$field * self.$field)*
            }
        }

        impl $VecN<f32> {
            #[inline]
            pub fn magnitude(&self) -> f32 {
                self.squared_magnitude().sqrt()
                // FIXME: once simd is part of std and stable, use it
                // rsqrt is faster than sqrt (but is approximate)
                // self.squared_magnitude().rsqrt()
            }
        }

        impl $VecN<f64> {
            #[inline]
            pub fn magnitude(&self) -> f64 {
                self.squared_magnitude().sqrt()
                // FIXME: once simd is part of std and stable, use it
                // rsqrt is faster than sqrt (but is approximate)
                // self.squared_magnitude().rsqrt()
            }
        }

        impl<F: Copy + Mul<F,Output=F>> Mul<F> for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn mul(self, rhs: F) -> $VecN<F> {
                $VecN {
                    $first: self.$first * rhs,
                    $($field: self.$field * rhs),*
                }
            }
        }
        impl Mul<$VecN<f32>> for f32 {
            type Output = $VecN<f32>;

            #[inline]
            fn mul(self, rhs: $VecN<f32>) -> $VecN<f32> {
                $VecN {
                    $first: self * rhs.$first,
                    $($field: self * rhs.$field),*
                }
            }
        }
        impl Mul<$VecN<f64>> for f64 {
            type Output = $VecN<f64>;

            #[inline]
            fn mul(self, rhs: $VecN<f64>) -> $VecN<f64> {
                $VecN {
                    $first: self * rhs.$first,
                    $($field: self * rhs.$field),*
                }
            }
        }

        impl<F: Copy + MulAssign<F>> MulAssign<F> for $VecN<F> {
            #[inline]
            fn mul_assign(&mut self, rhs: F) {
                self.$first *= rhs;
                $(self.$field *= rhs);*
            }
        }

        impl<F: Copy + Div<F,Output=F>> Div<F> for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn div(self, rhs: F) -> $VecN<F> {
                $VecN {
                    $first: self.$first / rhs,
                    $($field: self.$field / rhs),*
                }
            }
        }

        impl<F: Copy + DivAssign<F>> DivAssign<F> for $VecN<F> {
            #[inline]
            fn div_assign(&mut self, rhs: F) {
                self.$first /= rhs;
                $(self.$field /= rhs);*
            }
        }

        impl<F: Neg<Output=F>> Neg for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn neg(self) -> $VecN<F> {
                $VecN {
                    $first: -self.$first,
                    $($field: -self.$field),*
                }
            }
        }

        impl<F: Add<Output=F>> Add for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn add(self, other: $VecN<F>) -> $VecN<F> {
                $VecN {
                    $first: self.$first + other.$first,
                    $($field: self.$field + other.$field),*
                }
            }
        }

        impl<'a, F: Copy + AddAssign<F>> AddAssign<&'a $VecN<F>> for $VecN<F> {
            #[inline]
            fn add_assign(&mut self, other: &'a $VecN<F>) {
                self.$first += other.$first;
                $(self.$field += other.$field);*
            }
        }

        impl<F: Sub<Output=F>> Sub for $VecN<F> {
            type Output = $VecN<F>;

            #[inline]
            fn sub(self, other: $VecN<F>) -> $VecN<F> {
                $VecN {
                    $first: self.$first - other.$first,
                    $($field: self.$field - other.$field),*
                }
            }
        }

        impl<'a, F: Copy + SubAssign<F>> SubAssign<&'a $VecN<F>> for $VecN<F> {
            #[inline]
            fn sub_assign(&mut self, other: &'a $VecN<F>) {
                self.$first -= other.$first;
                $(self.$field -= other.$field);*
            }
        }

        impl<F: Copy + Mul<F,Output=F> + Add<F,Output=F>> $VecN<F> {
            #[inline]
            pub fn dot(&self, rhs: $VecN<F>) -> F {
                self.$first * rhs.$first
                    $(+ self.$field * rhs.$field)*
            }
        }

        impl<F: Copy + Mul<F,Output=F> + Div<F,Output=F> + Add<F,Output=F>> $VecN<F> {
            #[inline]
            pub fn project_onto(&self, axis: $VecN<F>) -> $VecN<F> {
                axis * (self.dot(axis) / axis.dot(axis))
            }
        }

        impl<F: Copy + Mul<F,Output=F> + Div<F,Output=F> + Add<F,Output=F> + Sub<F,Output=F>>
            $VecN<F>
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

impl<F: Copy + Mul<F,Output=F> + Sub<F,Output=F>> Vec3<F> {
    #[inline]
    pub fn cross(&self, rhs: Vec3<F>) -> Vec3<F> {
        Vec3::new(
            self.y*rhs.z - self.z*rhs.y,
            self.z*rhs.x - self.x*rhs.z,
            self.x*rhs.y - self.y*rhs.x
        )
    }
}

impl<F: Copy + Mul<F,Output=F> + Sub<F,Output=F> + Add<F,Output=F>> Vec3<F> {
    #[inline]
    pub fn triple_product(&self, b: Vec3<F>, c: Vec3<F>) -> F {
        self.cross(b).dot(c)
    }
}

// ----------------------------------------------------------------------------

impl<F> From<Vec4<F>> for Vec3<F> {
    fn from(v: Vec4<F>) -> Vec3<F> {
        Vec3 { x: v.x, y: v.y, z: v.z }
    }
}

impl<F> From<Vec3<F>> for Vec2<F> {
    fn from(v: Vec3<F>) -> Vec2<F> {
        Vec2 { x: v.x, y: v.y }
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use float_cmp:: ApproxEqUlps;
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
        assert!(VEC2.squared_magnitude().approx_eq_ulps(&5.0, 1));
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
