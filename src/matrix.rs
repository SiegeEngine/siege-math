
use num_traits::identities::{Zero, One};
use std::ops::{Index, IndexMut, Mul, Add, Neg};
use std::default::Default;
use super::{Vec2, Vec3, Vec4};

// NOTE: we store matrices in row-major order.  So Matrix.a is the first row (not column).

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

        impl<T: Default> Default for $MatN<T> {
            #[inline]
            fn default() -> $MatN<T> {
                $MatN { $first: $VecN::default(), $($field: $VecN::default()),* }
            }
        }

        impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<$MatN<T>> for $MatN<T> {
            type Output = $MatN<T>;

            #[inline]
            fn mul(self, rhs: $MatN<T>) -> $MatN<T> {
                $MatN {
                    $first: self.$first * rhs,
                    $($field: self.$field * rhs),*
                }
            }
        }
    }
}

impl_matrix!(Mat2 { a, b }, mat2, Vec2);
impl_matrix!(Mat3 { a, b, c }, mat3, Vec3);
impl_matrix!(Mat4 { a, b, c, d }, mat4, Vec4);


impl<T> Mat2<T> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.a.y, &mut self.b.x);
    }
}

impl<T> Mat3<T> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.a.y, &mut self.b.x);
        ::std::mem::swap(&mut self.a.z, &mut self.c.x);
        ::std::mem::swap(&mut self.b.z, &mut self.c.y);
    }
}

impl<T> Mat4<T> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.a.y, &mut self.b.x);
        ::std::mem::swap(&mut self.a.z, &mut self.c.x);
        ::std::mem::swap(&mut self.a.w, &mut self.d.x);
        ::std::mem::swap(&mut self.b.z, &mut self.c.y);
        ::std::mem::swap(&mut self.b.w, &mut self.d.y);
        ::std::mem::swap(&mut self.c.w, &mut self.d.z);
    }
}

impl<T: Zero + One> Mat2<T> {
    #[inline]
    pub fn identity() -> Mat2<T> {
        Mat2 {
            a: Vec2::new(T::one(), T::zero()),
            b: Vec2::new(T::zero(), T::one()),
        }
    }
}

impl<T: Zero + One> Mat3<T> {
    #[inline]
    pub fn identity() -> Mat3<T> {
        Mat3 {
            a: Vec3::new(T::one(), T::zero(), T::zero()),
            b: Vec3::new(T::zero(), T::one(), T::zero()),
            c: Vec3::new(T::zero(), T::zero(), T::one()),
        }
    }
}

impl<T: Zero + One> Mat4<T> {
    #[inline]
    pub fn identity() -> Mat4<T> {
        Mat4 {
            a: Vec4::new(T::one(), T::zero(), T::zero(), T::zero()),
            b: Vec4::new(T::zero(), T::one(), T::zero(), T::zero()),
            c: Vec4::new(T::zero(), T::zero(), T::one(), T::zero()),
            d: Vec4::new(T::zero(), T::zero(), T::zero(), T::one()),
        }
    }
}

impl<T: Zero + PartialEq> Mat2<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.a.y == T::zero() && self.b.x == T::zero()
    }
}

impl<T: Zero + PartialEq> Mat3<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.a.y == T::zero() && self.a.z == T::zero() &&
            self.b.x == T::zero() && self.b.z == T::zero() &&
            self.c.x == T::zero() && self.c.y == T::zero()
    }
}

impl<T: Zero + PartialEq> Mat4<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.a.y == T::zero() && self.a.z == T::zero() && self.a.w == T::zero() &&
            self.b.x == T::zero() && self.b.z == T::zero() && self.b.w == T::zero() &&
            self.c.x == T::zero() && self.c.y == T::zero() && self.c.w == T::zero() &&
            self.d.x == T::zero() && self.d.y == T::zero() && self.d.z == T::zero()
    }
}

impl<T: PartialEq> Mat2<T> {
    pub fn is_symmetric(&self) -> bool {
        self.a.y == self.b.x
    }
}


impl<T: PartialEq> Mat3<T> {
    pub fn is_symmetric(&self) -> bool {
        self.a.y == self.b.x &&
            self.a.z == self.c.x &&
            self.b.z == self.c.y
    }
}

impl<T: PartialEq> Mat4<T> {
    pub fn is_symmetric(&self) -> bool {
        self.a.y == self.b.x &&
            self.a.z == self.c.x &&
            self.a.w == self.d.x &&
            self.b.z == self.c.y &&
            self.b.w == self.d.y &&
            self.c.w == self.d.z
    }
}

impl<T: Copy + PartialEq + Neg<Output=T>> Mat2<T> {
    pub fn is_skew_symmetric(&self) -> bool {
        self.a.y == -self.b.x
    }
}

impl<T: Copy + PartialEq + Neg<Output=T>> Mat3<T> {
    pub fn is_skew_symmetric(&self) -> bool {
        self.a.y == -self.b.x &&
            self.a.z == -self.c.x &&
            self.b.z == -self.c.y
    }
}

impl<T: Copy + PartialEq + Neg<Output=T>> Mat4<T> {
    pub fn is_skew_symmetric(&self) -> bool {
        self.a.y == -self.b.x &&
            self.a.z == -self.c.x &&
            self.a.w == -self.d.x &&
            self.b.z == -self.c.y &&
            self.b.w == -self.d.y &&
            self.c.w == -self.d.z
    }
}


impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec2<T>> for Mat2<T> {
    type Output = Vec2<T>;

    #[inline]
    fn mul(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2::new(
            self.a.x * rhs.x
                + self.a.y * rhs.y,
            self.b.x * rhs.x
                + self.b.y * rhs.y
        )
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec3<T>> for Mat3<T> {
    type Output = Vec3<T>;

    #[inline]
    fn mul(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3::new(
            self.a.x * rhs.x
                + self.a.y * rhs.y
                + self.a.z * rhs.z,
            self.b.x * rhs.x
                + self.b.y * rhs.y
                + self.b.z * rhs.z,
            self.c.x * rhs.x
                + self.c.y * rhs.y
                + self.c.z * rhs.z,
        )
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec4<T>> for Mat4<T> {
    type Output = Vec4<T>;

    #[inline]
    fn mul(self, rhs: Vec4<T>) -> Vec4<T> {
        Vec4::new(
            self.a.x * rhs.x
                + self.a.y * rhs.y
                + self.a.z * rhs.z
                + self.a.w * rhs.w,
            self.b.x * rhs.x
                + self.b.y * rhs.y
                + self.b.z * rhs.z
                + self.b.w * rhs.w,
            self.c.x * rhs.x
                + self.c.y * rhs.y
                + self.c.z * rhs.z
                + self.c.w * rhs.w,
            self.d.x * rhs.x
                + self.d.y * rhs.y
                + self.d.z * rhs.z
                + self.d.w * rhs.w,
        )
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat2<T>> for Vec2<T> {
    type Output = Vec2<T>;

    #[inline]
    fn mul(self, rhs: Mat2<T>) -> Vec2<T> {
        Vec2::new(
            self.x * rhs.a.x
                + self.y * rhs.b.x,
            self.x * rhs.a.y
                + self.y * rhs.b.y
        )
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat3<T>> for Vec3<T> {
    type Output = Vec3<T>;

    #[inline]
    fn mul(self, rhs: Mat3<T>) -> Vec3<T> {
        Vec3::new(
            self.x * rhs.a.x
                + self.y * rhs.b.x
                + self.z * rhs.c.x,
            self.x * rhs.a.y
                + self.y * rhs.b.y
                + self.z * rhs.c.y,
            self.x * rhs.a.z
                + self.y * rhs.b.z
                + self.z * rhs.c.z
        )
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4<T>> for Vec4<T> {
    type Output = Vec4<T>;

    #[inline]
    fn mul(self, rhs: Mat4<T>) -> Vec4<T> {
        Vec4::new(
            self.x * rhs.a.x
                + self.y * rhs.b.x
                + self.z * rhs.c.x
                + self.w * rhs.d.x,
            self.x * rhs.a.y
                + self.y * rhs.b.y
                + self.z * rhs.c.y
                + self.w * rhs.d.y,
            self.x * rhs.a.z
                + self.y * rhs.b.z
                + self.z * rhs.c.z
                + self.w * rhs.d.z,
            self.x * rhs.a.w
                + self.y * rhs.b.w
                + self.z * rhs.c.w
                + self.w * rhs.d.w
        )
    }
}
