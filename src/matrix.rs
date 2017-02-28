
use num_traits::identities::{Zero, One};
use std::ops::{Index, IndexMut, Mul, Add, Neg, Div, Sub};
use std::default::Default;
use super::vector::{Vec2, Vec3, Vec4, vec2, vec3, vec4};

// NOTE: we store matrices in column-major order, which means we pre-multiply.
// This is traditional so matrices directly copied to the GPU will work with
// most other people's code, shaders, etc.  Also, it means vectors are stored
// contiguously in the matrix.
//
// However, we hide this internal storage format from the interface, and
// provide a row-major interface (e.g. via the [] operator and the new()
// function parameter order).  This way the programmer can write matrices
// in the same order that mathematicians write them.

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat2<T> {
    x: Vec2<T>,
    y: Vec2<T>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat3<T> {
    x: Vec3<T>,
    y: Vec3<T>,
    z: Vec3<T>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat4<T> {
    x: Vec4<T>,
    y: Vec4<T>,
    z: Vec4<T>,
    p: Vec4<T>
}

// -- impl Index --------------------------------------------------------------

// This is defined in row-major order
impl<T> Index<(usize,usize)> for Mat2<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row,col): (usize,usize)) -> &T {
        match col {
            0 => &self.x[row],
            1 => &self.y[row],
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<T> Index<(usize,usize)> for Mat3<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row,col): (usize,usize)) -> &T {
        match col {
            0 => &self.x[row],
            1 => &self.y[row],
            2 => &self.z[row],
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<T> Index<(usize,usize)> for Mat4<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row,col): (usize,usize)) -> &T {
        match col {
            0 => &self.x[row],
            1 => &self.y[row],
            2 => &self.z[row],
            3 => &self.p[row],
            _ => panic!("Index out of bounds for Mat4"),
        }
    }
}

// -- impl IndexMut -----------------------------------------------------------

impl<T> IndexMut<(usize,usize)> for Mat2<T> {
    #[inline]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut T {
        match col {
            0 => &mut self.x[row],
            1 => &mut self.y[row],
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<T> IndexMut<(usize,usize)> for Mat3<T> {
    #[inline]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut T {
        match col {
            0 => &mut self.x[row],
            1 => &mut self.y[row],
            2 => &mut self.z[row],
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<T> IndexMut<(usize,usize)> for Mat4<T> {
    #[inline]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut T {
        match col {
            0 => &mut self.x[row],
            1 => &mut self.y[row],
            2 => &mut self.z[row],
            3 => &mut self.p[row],
            _ => panic!("Index out of bounds for Mat4"),
        }
    }
}

// -- new ---------------------------------------------------------------------

impl<T> Mat2<T> {
    #[inline]
    pub fn new(r0c0: T, r0c1: T,
               r1c0: T, r1c1: T) -> Mat2<T>
    {
        Mat2 {
            x: vec2(r0c0, r1c0),
            y: vec2(r0c1, r1c1),
        }
    }

    #[inline]
    pub fn from_cols(x: Vec2<T>, y: Vec2<T>) -> Mat2<T>
    {
        Mat2 { x: x, y: y }
    }
}

impl<T> Mat3<T> {
    #[inline]
    pub fn new(r0c0: T, r0c1: T, r0c2: T,
               r1c0: T, r1c1: T, r1c2: T,
               r2c0: T, r2c1: T, r2c2: T) -> Mat3<T>
    {
        Mat3 {
            x: vec3(r0c0, r1c0, r2c0),
            y: vec3(r0c1, r1c1, r2c1),
            z: vec3(r0c2, r1c2, r2c2),
        }
    }

    #[inline]
    pub fn from_cols(x: Vec3<T>, y: Vec3<T>, z: Vec3<T>) -> Mat3<T>
    {
        Mat3 { x: x, y: y, z: z }
    }
}

impl<T> Mat4<T> {
    #[inline]
    pub fn new(r0c0: T, r0c1: T, r0c2: T, r0c3: T,
               r1c0: T, r1c1: T, r1c2: T, r1c3: T,
               r2c0: T, r2c1: T, r2c2: T, r2c3: T,
               r3c0: T, r3c1: T, r3c2: T, r3c3: T) -> Mat4<T>
    {
        Mat4 {
            x: vec4(r0c0, r1c0, r2c0, r3c0),
            y: vec4(r0c1, r1c1, r2c1, r3c1),
            z: vec4(r0c2, r1c2, r2c2, r3c2),
            p: vec4(r0c3, r1c3, r2c3, r3c3),
        }
    }

    #[inline]
    pub fn from_cols(x: Vec4<T>, y: Vec4<T>, z: Vec4<T>, p: Vec4<T>) -> Mat4<T>
    {
        Mat4 { x: x, y: y, z: z, p: p }
    }
}

// -- impl Default ------------------------------------------------------------

impl<T: Default> Default for Mat2<T> {
    #[inline]
    fn default() -> Mat2<T> {
        Mat2 {
            x: vec2(T::default(), T::default()),
            y: vec2(T::default(), T::default()),
        }
    }
}

impl<T: Default> Default for Mat3<T> {
    #[inline]
    fn default() -> Mat3<T> {
        Mat3 {
            x: vec3(T::default(), T::default(), T::default()),
            y: vec3(T::default(), T::default(), T::default()),
            z: vec3(T::default(), T::default(), T::default()),
        }
    }
}

impl<T: Default> Default for Mat4<T> {
    #[inline]
    fn default() -> Mat4<T> {
        Mat4 {
            x: vec4(T::default(), T::default(), T::default(), T::default()),
            y: vec4(T::default(), T::default(), T::default(), T::default()),
            z: vec4(T::default(), T::default(), T::default(), T::default()),
            p: vec4(T::default(), T::default(), T::default(), T::default()),
        }
    }
}

// -- zero --------------------------------------------------------------------

impl<T: Zero> Mat2<T> {
    #[inline]
    pub fn zero() -> Mat2<T>
    {
        Mat2 {
            x: vec2(T::zero(), T::zero()),
            y: vec2(T::zero(), T::zero()),
        }
    }
}

impl<T: Zero> Mat3<T> {
    #[inline]
    pub fn zero() -> Mat3<T>
    {
        Mat3 {
            x: vec3(T::zero(), T::zero(), T::zero()),
            y: vec3(T::zero(), T::zero(), T::zero()),
            z: vec3(T::zero(), T::zero(), T::zero()),
        }
    }
}

impl<T: Zero> Mat4<T> {
    #[inline]
    pub fn zero() -> Mat4<T>
    {
        Mat4 {
            x: vec4(T::zero(), T::zero(), T::zero(), T::zero()),
            y: vec4(T::zero(), T::zero(), T::zero(), T::zero()),
            z: vec4(T::zero(), T::zero(), T::zero(), T::zero()),
            p: vec4(T::zero(), T::zero(), T::zero(), T::zero()),
        }
    }
}

// -- identity ---------------------------------------------------------------

impl<T: Zero + One> Mat2<T> {
    #[inline]
    pub fn identity() -> Mat2<T>
    {
        Mat2 {
            x: vec2(T::one(), T::zero()),
            y: vec2(T::zero(), T::one()),
        }
    }
}

impl<T: Zero + One> Mat3<T> {
    #[inline]
    pub fn identity() -> Mat3<T>
    {
        Mat3 {
            x: vec3(T::one(), T::zero(), T::zero()),
            y: vec3(T::zero(), T::one(), T::zero()),
            z: vec3(T::zero(), T::zero(), T::one()),
        }
    }
}

impl<T: Zero + One> Mat4<T> {
    #[inline]
    pub fn identity() -> Mat4<T>
    {
        Mat4 {
            x: vec4(T::one(), T::zero(), T::zero(), T::zero()),
            y: vec4(T::zero(), T::one(), T::zero(), T::zero()),
            z: vec4(T::zero(), T::zero(), T::one(), T::zero()),
            p: vec4(T::zero(), T::zero(), T::zero(), T::one()),
        }
    }
}

// -- transpose ---------------------------------------------------------------

impl<T> Mat2<T> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.x.y, &mut self.y.x);
    }
}

impl<T> Mat3<T> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.x.y, &mut self.y.x);
        ::std::mem::swap(&mut self.x.z, &mut self.z.x);
        ::std::mem::swap(&mut self.y.z, &mut self.z.y);
    }
}

impl<T> Mat4<T> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.x.y, &mut self.y.x);
        ::std::mem::swap(&mut self.x.z, &mut self.z.x);
        ::std::mem::swap(&mut self.x.w, &mut self.p.x);
        ::std::mem::swap(&mut self.y.z, &mut self.z.y);
        ::std::mem::swap(&mut self.y.w, &mut self.p.y);
        ::std::mem::swap(&mut self.z.w, &mut self.p.z);
    }
}

// -- inverse -----------------------------------------------------------------

impl<T: Copy + One + Zero + PartialEq
     + Neg<Output=T> + Div<Output=T> + Sub<Output=T> + Mul<Output=T>> Mat2<T> {
    #[inline]
    pub fn determinant(&self) -> T {
        self.x.x * self.y.y - self.y.x * self.x.y
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat2<T>> {
        let d = self.determinant();
        if d == T::zero() { return None; }
        Some(Mat2 {
            x: vec2( self.y.y/d ,-self.x.y/d),
            y: vec2(-self.y.x/d , self.x.x/d),
        })
    }
}

impl<T: Copy + One + Zero + PartialEq
     + Neg<Output=T> + Div<Output=T> + Sub<Output=T> + Mul<Output=T>> Mat3<T> {
    #[inline]
    pub fn determinant(&self) -> T {
        self.x.x * (self.y.y * self.z.z - self.z.y * self.y.z)
            - self.y.x * (self.x.y * self.z.z - self.z.y * self.x.z)
            + self.z.x * (self.x.y * self.y.z - self.y.y * self.x.z)
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat3<T>> {
        let d = self.determinant();
        if d == T::zero() { return None; }
        let mut out = Mat3::from_cols(
            self.y.cross(self.z) / d,
            self.z.cross(self.x) / d,
            self.x.cross(self.y) / d);
        out.transpose();
        Some(out)
    }
}

impl<T: Copy + One + Zero + PartialEq
     + Neg<Output=T> + Div<Output=T> + Sub<Output=T> + Mul<Output=T>> Mat4<T> {
    #[inline]
    pub fn determinant(&self) -> T {
        self.x.x * Mat3 {
            x: vec3(self.y.y, self.y.z, self.y.w),
            y: vec3(self.z.y, self.z.z, self.z.w),
            z: vec3(self.p.y, self.p.z, self.p.w) }.determinant()

            - self.y.x * Mat3 {
                x: vec3(self.x.y, self.x.z, self.x.w),
                y: vec3(self.z.y, self.z.z, self.z.w),
                z: vec3(self.p.y, self.p.z, self.p.w) }.determinant()

            + self.z.x * Mat3 {
                x: vec3(self.x.y, self.x.z, self.x.w),
                y: vec3(self.y.y, self.y.z, self.y.w),
                z: vec3(self.p.y, self.p.z, self.p.w) }.determinant()

            - self.p.x * Mat3 {
                x: vec3(self.x.y, self.x.z, self.x.w),
                y: vec3(self.y.y, self.y.z, self.y.w),
                z: vec3(self.z.y, self.z.z, self.z.w) }.determinant()
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat4<T>> {
        let d = self.determinant();
        if d == T::zero() { return None; }
        let id = T::one() / d;
        let mut t = self.clone();
        t.transpose();
        let cf = |i, j| {
            let mat = match i {
                0 => Mat3::from_cols(t.y.truncate_n(j), t.z.truncate_n(j), t.p.truncate_n(j)),
                1 => Mat3::from_cols(t.x.truncate_n(j), t.z.truncate_n(j), t.p.truncate_n(j)),
                2 => Mat3::from_cols(t.x.truncate_n(j), t.y.truncate_n(j), t.p.truncate_n(j)),
                3 => Mat3::from_cols(t.x.truncate_n(j), t.y.truncate_n(j), t.z.truncate_n(j)),
                _ => panic!("out of range"),
            };
            let sign = if (i+j)&1 == 1 { -T::one() } else { T::one() };
            mat.determinant() * sign *id
        };

        Some(Mat4 {
            x: vec4(cf(0,0), cf(0,1), cf(0,2), cf(0,3)),
            y: vec4(cf(1,0), cf(1,1), cf(1,2), cf(1,3)),
            z: vec4(cf(2,0), cf(2,1), cf(2,2), cf(2,3)),
            p: vec4(cf(3,0), cf(3,1), cf(3,2), cf(3,3)),
        })
    }
}

// -- multiply by matrix ------------------------------------------------------

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'b Mat2<T>> for &'a Mat2<T> {
    type Output = Mat2<T>;

    #[inline]
    fn mul(self, rhs: &Mat2<T>) -> Mat2<T> {
        Mat2 {
            x: vec2( self.x.x * rhs.x.x + self.y.x * rhs.x.y,
                     self.x.y * rhs.x.x + self.y.y * rhs.x.y),
            y: vec2( self.x.x * rhs.y.x + self.y.x * rhs.y.y,
                     self.x.y * rhs.y.x + self.y.y * rhs.y.y),
        }
    }
}

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'b Mat3<T>> for &'a Mat3<T> {
    type Output = Mat3<T>;

    #[inline]
    fn mul(self, rhs: &Mat3<T>) -> Mat3<T> {
        Mat3 {
            x: vec3( self.x.x * rhs.x.x + self.y.x * rhs.x.y + self.z.x * rhs.x.z,
                     self.x.y * rhs.x.x + self.y.y * rhs.x.y + self.z.y * rhs.x.z,
                     self.x.z * rhs.x.x + self.y.z * rhs.x.y + self.z.z * rhs.x.z),
            y: vec3( self.x.x * rhs.y.x + self.y.x * rhs.y.y + self.z.x * rhs.y.z,
                     self.x.y * rhs.y.x + self.y.y * rhs.y.y + self.z.y * rhs.y.z,
                     self.x.z * rhs.y.x + self.y.z * rhs.y.y + self.z.z * rhs.y.z),
            z: vec3( self.x.x * rhs.z.x + self.y.x * rhs.z.y + self.z.x * rhs.z.z,
                     self.x.y * rhs.z.x + self.y.y * rhs.z.y + self.z.y * rhs.z.z,
                     self.x.z * rhs.z.x + self.y.z * rhs.z.y + self.z.z * rhs.z.z),
        }
    }
}

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'b Mat4<T>> for &'a Mat4<T> {
    type Output = Mat4<T>;

    #[inline]
    fn mul(self, rhs: &Mat4<T>) -> Mat4<T> {
        Mat4 {
            x: vec4( self.x.x * rhs.x.x + self.y.x * rhs.x.y + self.z.x * rhs.x.z + self.p.x * rhs.x.w,
                     self.x.y * rhs.x.x + self.y.y * rhs.x.y + self.z.y * rhs.x.z + self.p.y * rhs.x.w,
                     self.x.z * rhs.x.x + self.y.z * rhs.x.y + self.z.z * rhs.x.z + self.p.z * rhs.x.w,
                     self.x.w * rhs.x.x + self.y.w * rhs.x.y + self.z.w * rhs.x.z + self.p.w * rhs.x.w),
            y: vec4( self.x.x * rhs.y.x + self.y.x * rhs.y.y + self.z.x * rhs.y.z + self.p.x * rhs.y.w,
                     self.x.y * rhs.y.x + self.y.y * rhs.y.y + self.z.y * rhs.y.z + self.p.y * rhs.y.w,
                     self.x.z * rhs.y.x + self.y.z * rhs.y.y + self.z.z * rhs.y.z + self.p.z * rhs.y.w,
                     self.x.w * rhs.y.x + self.y.w * rhs.y.y + self.z.w * rhs.y.z + self.p.w * rhs.y.w),
            z: vec4( self.x.x * rhs.z.x + self.y.x * rhs.z.y + self.z.x * rhs.z.z + self.p.x * rhs.z.w,
                     self.x.y * rhs.z.x + self.y.y * rhs.z.y + self.z.y * rhs.z.z + self.p.y * rhs.z.w,
                     self.x.z * rhs.z.x + self.y.z * rhs.z.y + self.z.z * rhs.z.z + self.p.z * rhs.z.w,
                     self.x.w * rhs.z.x + self.y.w * rhs.z.y + self.z.w * rhs.z.z + self.p.w * rhs.z.w),
            p: vec4( self.x.x * rhs.p.x + self.y.x * rhs.p.y + self.z.x * rhs.p.z + self.p.x * rhs.p.w,
                     self.x.y * rhs.p.x + self.y.y * rhs.p.y + self.z.y * rhs.p.z + self.p.y * rhs.p.w,
                     self.x.z * rhs.p.x + self.y.z * rhs.p.y + self.z.z * rhs.p.z + self.p.z * rhs.p.w,
                     self.x.w * rhs.p.x + self.y.w * rhs.p.y + self.z.w * rhs.p.z + self.p.w * rhs.p.w)
        }
    }
}

// -- multiply by vector ------------------------------------------------------

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'a Vec2<T>> for &'b Mat2<T> {
    type Output = Vec2<T>;

    #[inline]
    fn mul(self, rhs: &Vec2<T>) -> Vec2<T> {
        Vec2::new( self.x.x * rhs.x + self.y.x * rhs.y,
                   self.x.y * rhs.x + self.y.y * rhs.y )
    }
}

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'a Vec3<T>> for &'b Mat3<T> {
    type Output = Vec3<T>;

    #[inline]
    fn mul(self, rhs: &Vec3<T>) -> Vec3<T> {
        Vec3::new( self.x.x * rhs.x + self.y.x * rhs.y + self.z.x * rhs.z,
                   self.x.y * rhs.x + self.y.y * rhs.y + self.z.y * rhs.z,
                   self.x.z * rhs.x + self.y.z * rhs.y + self.z.z * rhs.z)
    }
}

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'a Vec4<T>> for &'b Mat4<T> {
    type Output = Vec4<T>;

    #[inline]
    fn mul(self, rhs: &Vec4<T>) -> Vec4<T> {
        Vec4::new( self.x.x * rhs.x + self.y.x * rhs.y + self.z.x * rhs.z + self.p.x * rhs.w,
                   self.x.y * rhs.x + self.y.y * rhs.y + self.z.y * rhs.z + self.p.y * rhs.w,
                   self.x.z * rhs.x + self.y.z * rhs.y + self.z.z * rhs.z + self.p.z * rhs.w,
                   self.x.w * rhs.x + self.y.w * rhs.y + self.z.w * rhs.z + self.p.w * rhs.w)
    }
}

// -- multiply vector by matrix -----------------------------------------------

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'a Mat2<T>> for &'a Vec2<T> {
    type Output = Vec2<T>;

    #[inline]
    fn mul(self, rhs: &Mat2<T>) -> Vec2<T> {
        Vec2::new( self.x * rhs.x.x + self.y * rhs.x.y,
                   self.x * rhs.y.x + self.y * rhs.y.y )
    }
}

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'a Mat3<T>> for &'a Vec3<T> {
    type Output = Vec3<T>;

    #[inline]
    fn mul(self, rhs: &Mat3<T>) -> Vec3<T> {
        Vec3::new( self.x * rhs.x.x + self.y * rhs.x.y + self.z * rhs.x.z,
                   self.x * rhs.y.x + self.y * rhs.y.y + self.z * rhs.y.z,
                   self.x * rhs.z.x + self.y * rhs.z.y + self.z * rhs.z.z)
    }
}

impl<'a, 'b, T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<&'a Mat4<T>> for &'a Vec4<T> {
    type Output = Vec4<T>;

    #[inline]
    fn mul(self, rhs: &Mat4<T>) -> Vec4<T> {
        Vec4::new( self.x * rhs.x.x + self.y * rhs.x.y + self.z * rhs.x.z + self.w * rhs.x.w,
                   self.x * rhs.y.x + self.y * rhs.y.y + self.z * rhs.y.z + self.w * rhs.y.w,
                   self.x * rhs.z.x + self.y * rhs.z.y + self.z * rhs.z.z + self.w * rhs.z.w,
                   self.x * rhs.p.x + self.y * rhs.p.y + self.z * rhs.p.z + self.w * rhs.p.w)
    }
}

// -- characteristic tests ----------------------------------------------------


impl<T: Zero + PartialEq> Mat2<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.x.y == T::zero() && self.y.x == T::zero()
    }
}

impl<T: Zero + PartialEq> Mat3<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.x.y == T::zero() && self.x.z == T::zero() &&
            self.y.x == T::zero() && self.y.z == T::zero() &&
            self.z.x == T::zero() && self.z.y == T::zero()
    }
}

impl<T: Zero + PartialEq> Mat4<T> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.x.y == T::zero() && self.x.z == T::zero() && self.x.w == T::zero() &&
            self.y.x == T::zero() && self.y.z == T::zero() && self.y.w == T::zero() &&
            self.z.x == T::zero() && self.z.y == T::zero() && self.p.w == T::zero() &&
            self.p.x == T::zero() && self.p.y == T::zero() && self.z.z == T::zero()
    }
}

impl<T: PartialEq> Mat2<T> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.x.y == self.y.x
    }
}


impl<T: PartialEq> Mat3<T> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.x.y == self.y.x &&
            self.x.z == self.z.x &&
            self.y.z == self.z.y
    }
}

impl<T: PartialEq> Mat4<T> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.x.y == self.y.x &&
            self.x.z == self.z.x &&
            self.x.w == self.p.x &&
            self.y.z == self.z.y &&
            self.y.w == self.p.y &&
            self.z.w == self.p.z
    }
}

impl<T: Copy + PartialEq + Neg<Output=T>> Mat2<T> {
    #[inline]
    pub fn is_skew_symmetric(&self) -> bool {
        self.x.y == -self.y.x
    }
}

impl<T: Copy + PartialEq + Neg<Output=T>> Mat3<T> {
    #[inline]
    pub fn is_skew_symmetric(&self) -> bool {
        self.x.y == -self.y.x &&
            self.x.z == -self.z.x &&
            self.y.z == -self.z.y
    }
}

impl<T: Copy + PartialEq + Neg<Output=T>> Mat4<T> {
    #[inline]
    pub fn is_skew_symmetric(&self) -> bool {
        self.x.y == -self.y.x &&
            self.x.z == -self.z.x &&
            self.x.w == -self.p.x &&
            self.y.z == -self.z.y &&
            self.y.w == -self.p.y &&
            self.z.w == -self.p.z
    }
}


#[cfg(test)]
mod tests {
    use super::{Mat2, Mat3, Mat4};
    use super::super::vector::{Vec2, Vec3, Vec4};

    #[test]
    fn test_index() {
        let m: Mat2<u32> = Mat2::new(1, 2, 3, 4);
        assert_eq!(m[(0,0)], 1);
        assert_eq!(m[(0,1)], 2);
        assert_eq!(m[(1,0)], 3);
        assert_eq!(m[(1,1)], 4);
    }

    #[test]
    fn test_transpose() {
        let mut m: Mat2<u32> = Mat2::new(1, 2,
                                         3, 4);
        m.transpose();
        assert_eq!(m, Mat2::new(1, 3,
                                2, 4));

        let mut m: Mat3<u32> = Mat3::new(1, 2, 3,
                                         4, 5, 6,
                                         7, 8, 9);
        m.transpose();
        assert_eq!(m, Mat3::new(1, 4, 7,
                                2, 5, 8,
                                3, 6, 9));

        let mut m: Mat4<u32> = Mat4::new(1, 2, 3, 4,
                                         5, 6, 7, 8,
                                         9, 10, 11, 12,
                                         13, 14, 15, 16);
        m.transpose();
        assert_eq!(m, Mat4::new(1, 5, 9, 13,
                                2, 6, 10, 14,
                                3, 7, 11, 15,
                                4, 8, 12, 16));
    }

    #[test]
    fn test_mul_mat() {
        let left: Mat2<u32> = Mat2::new(1, 2, 3, 4);
        let right: Mat2<u32> = Mat2::new(6, 7, 8, 9);
        let product = &left * &right;
        assert_eq!(product, Mat2::new(22, 25,
                                      50, 57));

        let left: Mat3<u32> = Mat3::new(1, 2, 3,
                                        4, 5, 6,
                                        7, 8, 9);
        let right: Mat3<u32> = Mat3::new(10, 11, 12,
                                         13, 14, 15,
                                         16, 17, 18);
        let product = &left * &right;
        assert_eq!(product[(0,0)], 84);
        assert_eq!(product[(0,1)], 90);
        assert_eq!(product[(0,2)], 96);
        assert_eq!(product[(1,0)], 201);
        assert_eq!(product[(1,1)], 216);
        assert_eq!(product[(1,2)], 231);
        assert_eq!(product[(2,0)], 318);
        assert_eq!(product[(2,1)], 342);
        assert_eq!(product[(2,2)], 366);

        let left: Mat4<u32> = Mat4::new(1, 2, 3, 4,
                                        4, 3, 2, 1,
                                        5, 6, 2, 4,
                                        7, 1, 0, 3);
        let right: Mat4<u32> = Mat4::new(1, 6, 5, 2,
                                         3, 3, 3, 3,
                                         7, 8, 4, 1,
                                         9, 2, 0, 5);
        let product = &left * &right;
        assert_eq!(product[(0,0)], 64);
        assert_eq!(product[(0,1)], 44);
        assert_eq!(product[(0,2)], 23);
        assert_eq!(product[(0,3)], 31);
        assert_eq!(product[(1,0)], 36);
        assert_eq!(product[(1,1)], 51);
        assert_eq!(product[(1,2)], 37);
        assert_eq!(product[(1,3)], 24);
        assert_eq!(product[(2,0)], 73);
        assert_eq!(product[(2,1)], 72);
        assert_eq!(product[(2,2)], 51);
        assert_eq!(product[(2,3)], 50);
        assert_eq!(product[(3,0)], 37);
        assert_eq!(product[(3,1)], 51);
        assert_eq!(product[(3,2)], 38);
        assert_eq!(product[(3,3)], 32);
    }

    #[test]
    fn test_mul_vec() {
        let left: Mat2<u32> = Mat2::new(1, 2,
                                        3, 4);
        let right: Vec2<u32> = Vec2::new(10, 20);
        let product = &left * &right;
        assert_eq!(product[0], 50);
        assert_eq!(product[1], 110);
        let product = &right * &left;
        assert_eq!(product[0], 70);
        assert_eq!(product[1], 100);

        let left: Mat3<u32> = Mat3::new(1, 2, 3,
                                        4, 5, 6,
                                        7, 8, 9);
        let right: Vec3<u32> = Vec3::new(10, 20, 30);
        let product = &left * &right;
        assert_eq!(product[0], 140);
        assert_eq!(product[1], 320);
        assert_eq!(product[2], 500);
        let product = &right * &left;
        assert_eq!(product[0], 300);
        assert_eq!(product[1], 360);
        assert_eq!(product[2], 420);

        let left: Mat4<u32> = Mat4::new(1, 2, 3, 4,
                                        5, 6, 7, 8,
                                        9, 10, 11, 12,
                                        13, 14, 15, 16);
        let right: Vec4<u32> = Vec4::new(1, 2, 3, 4);
        let product = &left * &right;
        assert_eq!(product[0], 30);
        assert_eq!(product[1], 70);
        assert_eq!(product[2], 110);
        assert_eq!(product[3], 150);
        let product = &right * &left;
        assert_eq!(product[0], 90);
        assert_eq!(product[1], 100);
        assert_eq!(product[2], 110);
        assert_eq!(product[3], 120);
    }

    #[test]
    fn test_inverse() {
        assert_eq!(Mat2::<f64>::identity().inverse().unwrap(), Mat2::<f64>::identity());
        assert_eq!(Mat3::<f64>::identity().inverse().unwrap(), Mat3::<f64>::identity());
        assert_eq!(Mat4::<f64>::identity().inverse().unwrap(), Mat4::<f64>::identity());

        // This one works even with floating point inaccuracies.
        // But ideally we need ULPS comparison functions for vectors and matrices.
        let m = Mat3::new( 7.0, 2.0, 1.0,
                           0.0, 3.0, -1.0,
                           -3.0, 4.0, -2.0 );
        let inv = m.inverse().unwrap();
        assert_eq!(inv, Mat3::new( -2.0, 8.0, -5.0,
                                    3.0, -11.0, 7.0,
                                    9.0, -34.0, 21.0 ));

        // This one works even with floating point inaccuracies.
        // But ideally we need ULPS comparison functions for vectors and matrices.
        let m = Mat4::new( 1.0, 1.0, 1.0, 0.0,
                           0.0, 3.0, 1.0, 2.0,
                           2.0, 3.0, 1.0, 0.0,
                           1.0, 0.0, 2.0, 1.0 );
        let inv = m.inverse().unwrap();
        assert_eq!(inv, Mat4::new( -3.0, -0.5,   1.5,   1.0,
                                    1.0,  0.25, -0.25, -0.5,
                                    3.0,  0.25, -1.25, -0.5,
                                    -3.0, 0.0,   1.0,   1.0 ));
    }
}
