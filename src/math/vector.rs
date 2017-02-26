
use num_traits::identities::Zero;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
use std::default::Default;
use float_cmp::{ApproxEqRatio, ApproxEqUlps};


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

macro_rules! impl_vector {
    ($VecN:ident { $($field:ident),+ }, $constructor:ident) => {
        impl<T> $VecN<T> {
            /// Construct a new vector
            #[inline]
            pub fn new($($field: T),+) -> $VecN<T> {
                $VecN { $($field: $field),+ }
            }
        }

        /// Construct a new vector
        #[inline]
        pub fn $constructor<T>($($field: T),+) -> $VecN<T> {
            $VecN::new($($field),+)
        }

        impl<T: Zero> $VecN<T> {
            #[inline]
            pub fn zero() -> $VecN<T> {
                $VecN { $($field: T::zero()),+ }
            }
        }

        impl<T: Default> $VecN<T> {
            #[inline]
            pub fn default() -> $VecN<T> {
                $VecN { $($field: T::default()),+ }
            }
        }

        impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> $VecN<T>{
            #[inline]
            pub fn squared_magnitude(&self) -> T {
                $(self.$field * self.$field)++
            }
        }

        /*
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
*/
    }
}

impl_vector!(Vec2 { x, y }, vec2);
impl_vector!(Vec3 { x, y, z }, vec3);
impl_vector!(Vec4 { x, y, z, w }, vec4);

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


/*
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Vec3<T>{
    #[inline]
    pub fn squared_magnitude(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}
*/

