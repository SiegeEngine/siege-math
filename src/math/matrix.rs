
use super::{Vec2, Vec3, Vec4};
use num_traits::identities::Zero;
use std::ops::{Index, IndexMut};
use std::default::Default;

// NOTE: we store matrices in row-major order.  So Matrix.0 is the first row (not column).

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat2<T> {
    a: Vec2<T>,
    b: Vec2<T>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat3<T> {
    a: Vec3<T>,
    b: Vec3<T>,
    c: Vec3<T>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat4<T> {
    a: Vec4<T>,
    b: Vec4<T>,
    c: Vec4<T>,
    d: Vec4<T>
}

impl<T> Index<usize> for Mat2<T> {
    type Output = Vec2<T>;

    #[inline]
    fn index(&self, i: usize) -> &Vec2<T> {
        match i {
            0 => &self.a,
            1 => &self.b,
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<T> Index<usize> for Mat3<T> {
    type Output = Vec3<T>;

    #[inline]
    fn index(&self, i: usize) -> &Vec3<T> {
        match i {
            0 => &self.a,
            1 => &self.b,
            2 => &self.c,
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<T> Index<usize> for Mat4<T> {
    type Output = Vec4<T>;

    #[inline]
    fn index(&self, i: usize) -> &Vec4<T> {
        match i {
            0 => &self.a,
            1 => &self.b,
            2 => &self.c,
            3 => &self.d,
            _ => panic!("Index out of bounds for Mat4"),
        }
    }
}

impl<T> IndexMut<usize> for Mat2<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Vec2<T> {
        match i {
            0 => &mut self.a,
            1 => &mut self.b,
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<T> IndexMut<usize> for Mat3<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Vec3<T> {
        match i {
            0 => &mut self.a,
            1 => &mut self.b,
            2 => &mut self.c,
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<T> IndexMut<usize> for Mat4<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Vec4<T> {
        match i {
            0 => &mut self.a,
            1 => &mut self.b,
            2 => &mut self.c,
            3 => &mut self.d,
            _ => panic!("Index out of bounds for Mat4"),
        }
    }
}


macro_rules! impl_matrix {
    ($MatN:ident { $first:ident, $($field:ident),* }, $constructor:ident, $VecN:ident) => {
        impl<T> $MatN<T> {
            /// Construct a new matrix
            #[inline]
            pub fn new($first: $VecN<T>, $($field: $VecN<T>),*) -> $MatN<T> {
                $MatN { $first: $first,  $($field: $field),* }
            }
        }

        /// Construct a new matrix
        #[inline]
        pub fn $constructor<T>($first: $VecN<T>, $($field: $VecN<T>),*) -> $MatN<T> {
            $MatN::new($first, $($field),*)
        }

        impl<T: Zero> $MatN<T> {
            #[inline]
            pub fn zero() -> $MatN<T> {
                $MatN { $first: $VecN::zero(), $($field: $VecN::zero()),* }
            }
        }

        impl<T: Default> $MatN<T> {
            #[inline]
            pub fn default() -> $MatN<T> {
                $MatN { $first: $VecN::default(), $($field: $VecN::default()),* }
            }
        }
    }
}

impl_matrix!(Mat2 { a, b }, mat2, Vec2);
impl_matrix!(Mat3 { a, b, c }, mat3, Vec3);
impl_matrix!(Mat4 { a, b, c, d }, mat4, Vec4);

