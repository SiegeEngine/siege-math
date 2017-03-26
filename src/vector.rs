
use num_traits::identities::Zero;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
use std::default::Default;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Index<usize> for Vec2<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl<T> Index<usize> for Vec3<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<T> Index<usize> for Vec4<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl<T> IndexMut<usize> for Vec2<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl<T> IndexMut<usize> for Vec3<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<T> IndexMut<usize> for Vec4<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl<T: Copy> Vec3<T> {
    #[inline]
    pub fn truncate_n(&self, n: usize) -> Vec2<T> {
        match n {
            0 => vec2(self.y, self.z),
            1 => vec2(self.x, self.z),
            2 => vec2(self.x, self.y),
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<T: Copy> Vec4<T> {
    #[inline]
    pub fn truncate_n(&self, n: usize) -> Vec3<T> {
        match n {
            0 => vec3(self.y, self.z, self.w),
            1 => vec3(self.x, self.z, self.w),
            2 => vec3(self.x, self.y, self.w),
            3 => vec3(self.x, self.y, self.z),
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

macro_rules! impl_vector {
    ($VecN:ident { $first:ident, $($field:ident),* }, $constructor:ident) => {
        impl<T> $VecN<T> {
            /// Construct a new vector
            #[inline]
            pub fn new($first: T, $($field: T),*) -> $VecN<T> {
                $VecN { $first: $first, $($field: $field),* }
            }
        }

        /// Construct a new vector
        #[inline]
        pub fn $constructor<T>($first: T, $($field: T),*) -> $VecN<T> {
            $VecN::new($first, $($field),*)
        }

        impl<T: Zero> $VecN<T> {
            #[inline]
            pub fn zero() -> $VecN<T> {
                $VecN { $first: T::zero(), $($field: T::zero()),* }
            }
        }

        impl<T: Default> Default for $VecN<T> {
            #[inline]
            fn default() -> $VecN<T> {
                $VecN { $first: T::default(), $($field: T::default()),* }
            }
        }

        impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> $VecN<T>{
            #[inline]
            pub fn squared_magnitude(&self) -> T {
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

            #[inline]
            pub fn normalize(&mut self) {
                let mag = self.magnitude();
                self.$first /= mag;
                $(self.$field /= mag);*
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

            #[inline]
            pub fn normalize(&mut self) {
                let mag = self.magnitude();
                self.$first /= mag;
                $(self.$field /= mag);*
            }
        }

        impl<T: Copy + Mul<T,Output=T>> Mul<T> for $VecN<T> {
            type Output = $VecN<T>;

            #[inline]
            fn mul(self, rhs: T) -> $VecN<T> {
                $VecN {
                    $first: self.$first * rhs,
                    $($field: self.$field * rhs),*
                }
            }
        }

        impl<T: Copy + MulAssign<T>> MulAssign<T> for $VecN<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                self.$first *= rhs;
                $(self.$field *= rhs);*
            }
        }

        impl<T: Copy + Div<T,Output=T>> Div<T> for $VecN<T> {
            type Output = $VecN<T>;

            #[inline]
            fn div(self, rhs: T) -> $VecN<T> {
                $VecN {
                    $first: self.$first / rhs,
                    $($field: self.$field / rhs),*
                }
            }
        }

        impl<T: Copy + DivAssign<T>> DivAssign<T> for $VecN<T> {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                self.$first /= rhs;
                $(self.$field /= rhs);*
            }
        }

        impl<T: Neg<Output=T>> Neg for $VecN<T> {
            type Output = $VecN<T>;

            #[inline]
            fn neg(self) -> $VecN<T> {
                $VecN {
                    $first: -self.$first,
                    $($field: -self.$field),*
                }
            }
        }

        impl<T: Add<Output=T>> Add for $VecN<T> {
            type Output = $VecN<T>;

            #[inline]
            fn add(self, other: $VecN<T>) -> $VecN<T> {
                $VecN {
                    $first: self.$first + other.$first,
                    $($field: self.$field + other.$field),*
                }
            }
        }

        impl<'a, T: Copy + AddAssign<T>> AddAssign<&'a $VecN<T>> for $VecN<T> {
            #[inline]
            fn add_assign(&mut self, other: &'a $VecN<T>) {
                self.$first += other.$first;
                $(self.$field += other.$field);*
            }
        }

        impl<T: Sub<Output=T>> Sub for $VecN<T> {
            type Output = $VecN<T>;

            #[inline]
            fn sub(self, other: $VecN<T>) -> $VecN<T> {
                $VecN {
                    $first: self.$first - other.$first,
                    $($field: self.$field - other.$field),*
                }
            }
        }

        impl<'a, T: Copy + SubAssign<T>> SubAssign<&'a $VecN<T>> for $VecN<T> {
            #[inline]
            fn sub_assign(&mut self, other: &'a $VecN<T>) {
                self.$first -= other.$first;
                $(self.$field -= other.$field);*
            }
        }

        impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> $VecN<T> {
            #[inline]
            pub fn dot(&self, rhs: $VecN<T>) -> T {
                self.$first * rhs.$first
                    $(+ self.$field * rhs.$field)*
            }
        }
    }
}

impl_vector!(Vec2 { x, y }, vec2);
impl_vector!(Vec3 { x, y, z }, vec3);
impl_vector!(Vec4 { x, y, z, w }, vec4);

impl<T: Copy + Mul<T,Output=T> + Sub<T,Output=T>> Vec3<T> {
    #[inline]
    pub fn cross(&self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3::new(
            self.y*rhs.z - self.z*rhs.y,
            self.z*rhs.x - self.x*rhs.z,
            self.x*rhs.y - self.y*rhs.x
        )
    }
}

impl<T: Copy + Mul<T,Output=T> + Sub<T,Output=T> + Add<T,Output=T>> Vec3<T> {
    #[inline]
    pub fn triple_product(&self, b: Vec3<T>, c: Vec3<T>) -> T {
        self.cross(b).dot(c)
    }
}

impl<T: Copy + Mul<T,Output=T> + Div<T,Output=T> + Add<T,Output=T>> Vec3<T> {
    pub fn project_onto(&self, axis: Vec3<T>) -> Vec3<T> {
        axis * (self.dot(axis) / axis.squared_magnitude())
    }
}

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
