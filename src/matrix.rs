
use num_traits::identities::{Zero, One};
use num_traits::float::Float;
use std::ops::{Index, IndexMut, Mul, Add, Neg, Div, Sub};
use std::default::Default;
use super::vector::{Vec2, Vec3, Vec4, Direction3, Point3};
use super::Angle;

// NOTE: we store matrices in column-major order, which means we pre-multiply.
// This is traditional so matrices directly copied to the GPU will work with
// most other people's code, shaders, etc.  Also, it means vectors are stored
// contiguously in the matrix.
//
// However, we hide this internal storage format from the interface, and
// provide a row-major interface (e.g. via the [] operator and the new()
// function parameter order).  This way the programmer can write matrices
// in the same order that mathematicians write them.
// (Subsequently we have made the internals public so that we can define
//  constant matrices).

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat2<F> {
    pub x: Vec2<F>,
    pub y: Vec2<F>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat3<F> {
    pub x: Vec3<F>,
    pub y: Vec3<F>,
    pub z: Vec3<F>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Mat4<F> {
    pub x: Vec4<F>,
    pub y: Vec4<F>,
    pub z: Vec4<F>,
    pub p: Vec4<F>
}

// -- impl Index --------------------------------------------------------------

// This is defined in row-major order
impl<F> Index<(usize,usize)> for Mat2<F> {
    type Output = F;

    #[inline]
    fn index(&self, (row,col): (usize,usize)) -> &F {
        match col {
            0 => &self.x[row],
            1 => &self.y[row],
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<F> Index<(usize,usize)> for Mat3<F> {
    type Output = F;

    #[inline]
    fn index(&self, (row,col): (usize,usize)) -> &F {
        match col {
            0 => &self.x[row],
            1 => &self.y[row],
            2 => &self.z[row],
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<F> Index<(usize,usize)> for Mat4<F> {
    type Output = F;

    #[inline]
    fn index(&self, (row,col): (usize,usize)) -> &F {
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

impl<F> IndexMut<(usize,usize)> for Mat2<F> {
    #[inline]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut F {
        match col {
            0 => &mut self.x[row],
            1 => &mut self.y[row],
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<F> IndexMut<(usize,usize)> for Mat3<F> {
    #[inline]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut F {
        match col {
            0 => &mut self.x[row],
            1 => &mut self.y[row],
            2 => &mut self.z[row],
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<F> IndexMut<(usize,usize)> for Mat4<F> {
    #[inline]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut F {
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

impl<F> Mat2<F> {
    #[inline]
    pub fn new(r0c0: F, r0c1: F,
               r1c0: F, r1c1: F) -> Mat2<F>
    {
        Mat2 { // looks transposed because stored column-major
            x: Vec2::new(r0c0, r1c0),
            y: Vec2::new(r0c1, r1c1),
        }
    }

    #[inline]
    pub fn from_cols(x: Vec2<F>, y: Vec2<F>) -> Mat2<F>
    {
        Mat2 { x: x, y: y }
    }
}

impl<F> Mat3<F> {
    #[inline]
    pub fn new(r0c0: F, r0c1: F, r0c2: F,
               r1c0: F, r1c1: F, r1c2: F,
               r2c0: F, r2c1: F, r2c2: F) -> Mat3<F>
    {
        Mat3 { // looks transposed because stored column-major
            x: Vec3::new(r0c0, r1c0, r2c0),
            y: Vec3::new(r0c1, r1c1, r2c1),
            z: Vec3::new(r0c2, r1c2, r2c2),
        }
    }

    #[inline]
    pub fn from_cols(x: Vec3<F>, y: Vec3<F>, z: Vec3<F>) -> Mat3<F>
    {
        Mat3 { x: x, y: y, z: z }
    }
}

impl<F> Mat4<F> {
    #[inline]
    pub fn new(r0c0: F, r0c1: F, r0c2: F, r0c3: F,
               r1c0: F, r1c1: F, r1c2: F, r1c3: F,
               r2c0: F, r2c1: F, r2c2: F, r2c3: F,
               r3c0: F, r3c1: F, r3c2: F, r3c3: F) -> Mat4<F>
    {
        Mat4 { // looks transposed because stored column-major
            x: Vec4::new(r0c0, r1c0, r2c0, r3c0),
            y: Vec4::new(r0c1, r1c1, r2c1, r3c1),
            z: Vec4::new(r0c2, r1c2, r2c2, r3c2),
            p: Vec4::new(r0c3, r1c3, r2c3, r3c3),
        }
    }

    #[inline]
    pub fn from_cols(x: Vec4<F>, y: Vec4<F>, z: Vec4<F>, p: Vec4<F>) -> Mat4<F>
    {
        Mat4 { x: x, y: y, z: z, p: p }
    }
}

impl<F: Zero + One> Mat4<F> {
    #[inline]
    pub fn from_components(x_dir: Direction3<F>, y_dir: Direction3<F>, z_dir: Direction3<F>,
                           pos: Point3<F>) -> Mat4<F>
    {
        Mat4 {  // looks transposed because stored column-major
            x: Vec4::new(x_dir.0.x, x_dir.0.y, x_dir.0.z, F::zero()),
            y: Vec4::new(y_dir.0.x, y_dir.0.y, y_dir.0.z, F::zero()),
            z: Vec4::new(z_dir.0.x, z_dir.0.y, z_dir.0.z, F::zero()),
            p: Vec4::new(  pos.0.x,   pos.0.y,   pos.0.z, F::one())
        }
    }

    #[inline]
    pub fn from_mat3(mat3: Mat3<F>, pos: Point3<F>) -> Mat4<F>
    {
        Mat4 {  // looks transposed because stored column-major
            x: Vec4::new(mat3.x.x, mat3.x.y, mat3.x.z, F::zero()),
            y: Vec4::new(mat3.y.x, mat3.y.y, mat3.y.z, F::zero()),
            z: Vec4::new(mat3.z.x, mat3.z.y, mat3.z.z, F::zero()),
            p: Vec4::new( pos.0.x,  pos.0.y,  pos.0.z, F::one())
        }
    }
}

// -- impl Default ------------------------------------------------------------

impl<F: Default> Default for Mat2<F> {
    #[inline]
    fn default() -> Mat2<F> {
        Mat2::new( F::default(), F::default(),
                   F::default(), F::default() )
    }
}

impl<F: Default> Default for Mat3<F> {
    #[inline]
    fn default() -> Mat3<F> {
        Mat3::new( F::default(), F::default(), F::default(),
                   F::default(), F::default(), F::default(),
                   F::default(), F::default(), F::default() )
    }
}

impl<F: Default> Default for Mat4<F> {
    #[inline]
    fn default() -> Mat4<F> {
        Mat4::new( F::default(), F::default(), F::default(), F::default(),
                   F::default(), F::default(), F::default(), F::default(),
                   F::default(), F::default(), F::default(), F::default(),
                   F::default(), F::default(), F::default(), F::default() )
    }
}

// -- zero --------------------------------------------------------------------

impl<F: Zero> Mat2<F> {
    #[inline]
    pub fn zero() -> Mat2<F>
    {
        Mat2::new( F::zero(), F::zero(),
                   F::zero(), F::zero() )
    }
}

impl<F: Zero> Mat3<F> {
    #[inline]
    pub fn zero() -> Mat3<F>
    {
        Mat3::new( F::zero(), F::zero(), F::zero(),
                   F::zero(), F::zero(), F::zero(),
                   F::zero(), F::zero(), F::zero() )
    }
}

impl<F: Zero> Mat4<F> {
    #[inline]
    pub fn zero() -> Mat4<F>
    {
        Mat4::new( F::zero(), F::zero(), F::zero(), F::zero(),
                   F::zero(), F::zero(), F::zero(), F::zero(),
                   F::zero(), F::zero(), F::zero(), F::zero(),
                   F::zero(), F::zero(), F::zero(), F::zero() )
    }
}

// -- identity ---------------------------------------------------------------

impl<F: Zero + One> Mat2<F> {
    #[inline]
    pub fn identity() -> Mat2<F>
    {
        Mat2::new( F::one(), F::zero(),
                   F::zero(), F::one() )
    }
}

impl<F: Zero + One> Mat3<F> {
    #[inline]
    pub fn identity() -> Mat3<F>
    {
        Mat3::new( F::one(), F::zero(), F::zero(),
                   F::zero(), F::one(), F::zero(),
                   F::zero(), F::zero(), F::one() )
    }
}

impl<F: Zero + One> Mat4<F> {
    #[inline]
    pub fn identity() -> Mat4<F>
    {
        Mat4::new( F::one(), F::zero(), F::zero(), F::zero(),
                   F::zero(), F::one(), F::zero(), F::zero(),
                   F::zero(), F::zero(), F::one(), F::zero(),
                   F::zero(), F::zero(), F::zero(), F::one() )
    }
}

// -- transpose ---------------------------------------------------------------

impl<F> Mat2<F> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.x.y, &mut self.y.x);
    }
}

impl<F> Mat3<F> {
    #[inline]
    pub fn transpose(&mut self) {
        ::std::mem::swap(&mut self.x.y, &mut self.y.x);
        ::std::mem::swap(&mut self.x.z, &mut self.z.x);
        ::std::mem::swap(&mut self.y.z, &mut self.z.y);
    }
}

impl<F> Mat4<F> {
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

impl<F: Copy + One + Zero + PartialEq
     + Neg<Output=F> + Div<Output=F> + Sub<Output=F> + Mul<Output=F>> Mat2<F> {
    #[inline]
    pub fn determinant(&self) -> F {
        self.x.x * self.y.y - self.y.x * self.x.y
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat2<F>> {
        let d = self.determinant();
        if d == F::zero() { return None; }
        Some(Mat2::new( self.y.y/d, -self.y.x/d,
                        -self.x.y/d, self.x.x/d ))
    }
}

impl<F: Copy + One + Zero + PartialEq
     + Neg<Output=F> + Div<Output=F> + Sub<Output=F> + Mul<Output=F>> Mat3<F> {
    #[inline]
    pub fn determinant(&self) -> F {
        self.x.x * (self.y.y * self.z.z - self.z.y * self.y.z)
            - self.y.x * (self.x.y * self.z.z - self.z.y * self.x.z)
            + self.z.x * (self.x.y * self.y.z - self.y.y * self.x.z)
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat3<F>> {
        let d = self.determinant();
        if d == F::zero() { return None; }
        let mut out = Mat3::from_cols(
            self.y.cross(self.z) / d,
            self.z.cross(self.x) / d,
            self.x.cross(self.y) / d);
        out.transpose();
        Some(out)
    }
}

impl<F: Copy + One + Zero + PartialEq
     + Neg<Output=F> + Div<Output=F> + Sub<Output=F> + Mul<Output=F>> Mat4<F> {
    #[inline]
    pub fn determinant(&self) -> F {
        self.x.x * Mat3::new( self.y.y, self.z.y, self.p.y,
                              self.y.z, self.z.z, self.p.z,
                              self.y.w, self.z.w, self.p.w ).determinant()
            - self.y.x * Mat3::new( self.x.y, self.z.y, self.p.y,
                                    self.x.z, self.z.z, self.p.z,
                                    self.x.w, self.z.w, self.p.w ).determinant()
            + self.z.x * Mat3::new( self.x.y, self.y.y, self.p.y,
                                    self.x.z, self.y.z, self.p.z,
                                    self.x.w, self.y.w, self.p.w ).determinant()
            - self.p.x * Mat3::new( self.x.y, self.y.y, self.z.y,
                                    self.x.z, self.y.z, self.z.z,
                                    self.x.w, self.y.w, self.z.w ).determinant()
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat4<F>> {
        let d = self.determinant();
        if d == F::zero() { return None; }
        let id = F::one() / d;
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
            let sign = if (i+j)&1 == 1 { -F::one() } else { F::one() };
            mat.determinant() * sign *id
        };

        Some(Mat4::new( cf(0,0), cf(1,0), cf(2,0), cf(3,0),
                        cf(0,1), cf(1,1), cf(2,1), cf(3,1),
                        cf(0,2), cf(1,2), cf(2,2), cf(3,2),
                        cf(0,3), cf(1,3), cf(2,3), cf(3,3) ))
    }
}

// -- add ---------------------------------------------------------------------

impl<'a, 'b, F: Copy + Add<F,Output=F>> Add<&'b Mat2<F>> for &'a Mat2<F> {
    type Output = Mat2<F>;

    #[inline]
    fn add(self, rhs: &Mat2<F>) -> Mat2<F> {
        Mat2::new( self.x.x + rhs.x.x,  self.y.x + rhs.y.x,
                   self.x.y + rhs.x.y,  self.y.y + rhs.y.y )
    }
}

impl<'a, 'b, F: Copy + Add<F,Output=F>> Add<&'b Mat3<F>> for &'a Mat3<F> {
    type Output = Mat3<F>;

    #[inline]
    fn add(self, rhs: &Mat3<F>) -> Mat3<F> {
        Mat3::new( self.x.x + rhs.x.x,  self.y.x + rhs.y.x,  self.z.x + rhs.z.x,
                   self.x.y + rhs.x.y,  self.y.y + rhs.y.y,  self.z.y + rhs.z.y,
                   self.x.z + rhs.x.z,  self.y.z + rhs.y.z,  self.z.z + rhs.z.z )
    }
}

impl<'a, 'b, F: Copy + Add<F,Output=F>> Add<&'b Mat4<F>> for &'a Mat4<F> {
    type Output = Mat4<F>;

    #[inline]
    fn add(self, rhs: &Mat4<F>) -> Mat4<F> {
        Mat4::new(
            self.x.x + rhs.x.x, self.y.x + rhs.y.x, self.z.x + rhs.z.x, self.p.x + rhs.p.x,
            self.x.y + rhs.x.y, self.y.y + rhs.y.y, self.z.y + rhs.z.y, self.p.y + rhs.p.y,
            self.x.z + rhs.x.z, self.y.z + rhs.y.z, self.z.z + rhs.z.z, self.p.z + rhs.p.z,
            self.x.w + rhs.x.w, self.y.w + rhs.y.z, self.z.w + rhs.z.w, self.p.w + rhs.p.w )
    }
}

// -- multiply by scalar ------------------------------------------------------

impl<'a, F: Copy + Mul<F,Output=F>> Mul<F> for &'a Mat2<F> {
    type Output = Mat2<F>;

    #[inline]
    fn mul(self, rhs: F) -> Mat2<F> {
        Mat2::new( self.x.x * rhs, self.y.x * rhs,
                   self.x.y * rhs, self.y.y * rhs )
    }
}

impl<'a, F: Copy + Mul<F,Output=F>> Mul<F> for &'a Mat3<F> {
    type Output = Mat3<F>;

    #[inline]
    fn mul(self, rhs: F) -> Mat3<F> {
        Mat3::new( self.x.x * rhs, self.y.x * rhs, self.z.x * rhs,
                   self.x.y * rhs, self.y.y * rhs, self.z.y * rhs,
                   self.x.z * rhs, self.y.z * rhs, self.z.z * rhs )
    }
}

impl<'a, F: Copy + Mul<F,Output=F>> Mul<F> for &'a Mat4<F> {
    type Output = Mat4<F>;

    #[inline]
    fn mul(self, rhs: F) -> Mat4<F> {
        Mat4::new( self.x.x * rhs, self.y.x * rhs, self.z.x * rhs, self.p.x * rhs,
                   self.x.y * rhs, self.y.y * rhs, self.z.y * rhs, self.p.y * rhs,
                   self.x.z * rhs, self.y.z * rhs, self.z.z * rhs, self.p.z * rhs,
                   self.x.w * rhs, self.y.w * rhs, self.z.w * rhs, self.p.w * rhs )
    }
}

// -- multiply by matrix ------------------------------------------------------

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'b Mat2<F>> for &'a Mat2<F> {
    type Output = Mat2<F>;

    #[inline]
    fn mul(self, rhs: &Mat2<F>) -> Mat2<F> {
        Mat2::new(
            self.x.x * rhs.x.x + self.y.x * rhs.x.y,  self.x.x * rhs.y.x + self.y.x * rhs.y.y,
            self.x.y * rhs.x.x + self.y.y * rhs.x.y,  self.x.y * rhs.y.x + self.y.y * rhs.y.y)
    }
}

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'b Mat3<F>> for &'a Mat3<F> {
    type Output = Mat3<F>;

    #[inline]
    fn mul(self, rhs: &Mat3<F>) -> Mat3<F> {
        Mat3::new(
            self.x.x * rhs.x.x + self.y.x * rhs.x.y + self.z.x * rhs.x.z,
            self.x.x * rhs.y.x + self.y.x * rhs.y.y + self.z.x * rhs.y.z,
            self.x.x * rhs.z.x + self.y.x * rhs.z.y + self.z.x * rhs.z.z,

            self.x.y * rhs.x.x + self.y.y * rhs.x.y + self.z.y * rhs.x.z,
            self.x.y * rhs.y.x + self.y.y * rhs.y.y + self.z.y * rhs.y.z,
            self.x.y * rhs.z.x + self.y.y * rhs.z.y + self.z.y * rhs.z.z,

            self.x.z * rhs.x.x + self.y.z * rhs.x.y + self.z.z * rhs.x.z,
            self.x.z * rhs.y.x + self.y.z * rhs.y.y + self.z.z * rhs.y.z,
            self.x.z * rhs.z.x + self.y.z * rhs.z.y + self.z.z * rhs.z.z)
    }
}

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'b Mat4<F>> for &'a Mat4<F> {
    type Output = Mat4<F>;

    #[inline]
    fn mul(self, rhs: &Mat4<F>) -> Mat4<F> {
        Mat4::new(
            self.x.x * rhs.x.x + self.y.x * rhs.x.y + self.z.x * rhs.x.z + self.p.x * rhs.x.w,
            self.x.x * rhs.y.x + self.y.x * rhs.y.y + self.z.x * rhs.y.z + self.p.x * rhs.y.w,
            self.x.x * rhs.z.x + self.y.x * rhs.z.y + self.z.x * rhs.z.z + self.p.x * rhs.z.w,
            self.x.x * rhs.p.x + self.y.x * rhs.p.y + self.z.x * rhs.p.z + self.p.x * rhs.p.w,

            self.x.y * rhs.x.x + self.y.y * rhs.x.y + self.z.y * rhs.x.z + self.p.y * rhs.x.w,
            self.x.y * rhs.y.x + self.y.y * rhs.y.y + self.z.y * rhs.y.z + self.p.y * rhs.y.w,
            self.x.y * rhs.z.x + self.y.y * rhs.z.y + self.z.y * rhs.z.z + self.p.y * rhs.z.w,
            self.x.y * rhs.p.x + self.y.y * rhs.p.y + self.z.y * rhs.p.z + self.p.y * rhs.p.w,

            self.x.z * rhs.x.x + self.y.z * rhs.x.y + self.z.z * rhs.x.z + self.p.z * rhs.x.w,
            self.x.z * rhs.y.x + self.y.z * rhs.y.y + self.z.z * rhs.y.z + self.p.z * rhs.y.w,
            self.x.z * rhs.z.x + self.y.z * rhs.z.y + self.z.z * rhs.z.z + self.p.z * rhs.z.w,
            self.x.z * rhs.p.x + self.y.z * rhs.p.y + self.z.z * rhs.p.z + self.p.z * rhs.p.w,

            self.x.w * rhs.x.x + self.y.w * rhs.x.y + self.z.w * rhs.x.z + self.p.w * rhs.x.w,
            self.x.w * rhs.y.x + self.y.w * rhs.y.y + self.z.w * rhs.y.z + self.p.w * rhs.y.w,
            self.x.w * rhs.z.x + self.y.w * rhs.z.y + self.z.w * rhs.z.z + self.p.w * rhs.z.w,
            self.x.w * rhs.p.x + self.y.w * rhs.p.y + self.z.w * rhs.p.z + self.p.w * rhs.p.w )
    }
}

// -- multiply by vector ------------------------------------------------------

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'a Vec2<F>> for &'b Mat2<F> {
    type Output = Vec2<F>;

    #[inline]
    fn mul(self, rhs: &Vec2<F>) -> Vec2<F> {
        Vec2::new( self.x.x * rhs.x + self.y.x * rhs.y,
                   self.x.y * rhs.x + self.y.y * rhs.y )
    }
}

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'a Vec3<F>> for &'b Mat3<F> {
    type Output = Vec3<F>;

    #[inline]
    fn mul(self, rhs: &Vec3<F>) -> Vec3<F> {
        Vec3::new( self.x.x * rhs.x + self.y.x * rhs.y + self.z.x * rhs.z,
                   self.x.y * rhs.x + self.y.y * rhs.y + self.z.y * rhs.z,
                   self.x.z * rhs.x + self.y.z * rhs.y + self.z.z * rhs.z)
    }
}

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'a Vec4<F>> for &'b Mat4<F> {
    type Output = Vec4<F>;

    #[inline]
    fn mul(self, rhs: &Vec4<F>) -> Vec4<F> {
        Vec4::new( self.x.x * rhs.x + self.y.x * rhs.y + self.z.x * rhs.z + self.p.x * rhs.w,
                   self.x.y * rhs.x + self.y.y * rhs.y + self.z.y * rhs.z + self.p.y * rhs.w,
                   self.x.z * rhs.x + self.y.z * rhs.y + self.z.z * rhs.z + self.p.z * rhs.w,
                   self.x.w * rhs.x + self.y.w * rhs.y + self.z.w * rhs.z + self.p.w * rhs.w)
    }
}

// -- multiply vector by matrix -----------------------------------------------

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'a Mat2<F>> for &'a Vec2<F> {
    type Output = Vec2<F>;

    #[inline]
    fn mul(self, rhs: &Mat2<F>) -> Vec2<F> {
        Vec2::new( self.x * rhs.x.x + self.y * rhs.x.y,
                   self.x * rhs.y.x + self.y * rhs.y.y )
    }
}

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'a Mat3<F>> for &'a Vec3<F> {
    type Output = Vec3<F>;

    #[inline]
    fn mul(self, rhs: &Mat3<F>) -> Vec3<F> {
        Vec3::new( self.x * rhs.x.x + self.y * rhs.x.y + self.z * rhs.x.z,
                   self.x * rhs.y.x + self.y * rhs.y.y + self.z * rhs.y.z,
                   self.x * rhs.z.x + self.y * rhs.z.y + self.z * rhs.z.z)
    }
}

impl<'a, 'b, F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Mul<&'a Mat4<F>> for &'a Vec4<F> {
    type Output = Vec4<F>;

    #[inline]
    fn mul(self, rhs: &Mat4<F>) -> Vec4<F> {
        Vec4::new( self.x * rhs.x.x + self.y * rhs.x.y + self.z * rhs.x.z + self.w * rhs.x.w,
                   self.x * rhs.y.x + self.y * rhs.y.y + self.z * rhs.y.z + self.w * rhs.y.w,
                   self.x * rhs.z.x + self.y * rhs.z.y + self.z * rhs.z.z + self.w * rhs.z.w,
                   self.x * rhs.p.x + self.y * rhs.p.y + self.z * rhs.p.z + self.w * rhs.p.w)
    }
}

// -- characteristic tests ----------------------------------------------------

impl<F: Zero + PartialEq> Mat2<F> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.x.y == F::zero() && self.y.x == F::zero()
    }
}

impl<F: Zero + PartialEq> Mat3<F> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.x.y == F::zero() && self.x.z == F::zero() &&
            self.y.x == F::zero() && self.y.z == F::zero() &&
            self.z.x == F::zero() && self.z.y == F::zero()
    }
}

impl<F: Zero + PartialEq> Mat4<F> {
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        self.x.y == F::zero() && self.x.z == F::zero() && self.x.w == F::zero() &&
            self.y.x == F::zero() && self.y.z == F::zero() && self.y.w == F::zero() &&
            self.z.x == F::zero() && self.z.y == F::zero() && self.p.w == F::zero() &&
            self.p.x == F::zero() && self.p.y == F::zero() && self.z.z == F::zero()
    }
}

impl<F: PartialEq> Mat2<F> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.x.y == self.y.x
    }
}


impl<F: PartialEq> Mat3<F> {
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.x.y == self.y.x &&
            self.x.z == self.z.x &&
            self.y.z == self.z.y
    }
}

impl<F: PartialEq> Mat4<F> {
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

impl<F: Copy + PartialEq + Neg<Output=F>> Mat2<F> {
    #[inline]
    pub fn is_skew_symmetric(&self) -> bool {
        self.x.y == -self.y.x
    }
}

impl<F: Copy + PartialEq + Neg<Output=F>> Mat3<F> {
    #[inline]
    pub fn is_skew_symmetric(&self) -> bool {
        self.x.y == -self.y.x &&
            self.x.z == -self.z.x &&
            self.y.z == -self.z.y
    }
}

impl<F: Copy + PartialEq + Neg<Output=F>> Mat4<F> {
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

// -- mat4 components ---------------------------------------------------------

impl<F: Copy> Mat4<F> {
    pub fn get_translation(&self) -> Point3<F>
    {
        Point3(self.p.truncate_w())
    }
}

impl<F: One> Mat4<F> {
    pub fn set_translation(&mut self, p: Point3<F>)
    {
        self.p = From::from(p);
    }
}

// -- rotation matrices -------------------------------------------------------

impl<F: Float> Mat2<F> {
    #[inline]
    pub fn from_angle(theta: Angle<F>) -> Mat2<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat2 {
            x: Vec2::new(c, s),
            y: Vec2::new(-s, c),
        }
    }
}

impl<F: Float + One + Zero> Mat3<F> {
    #[inline]
    pub fn from_angle_x(theta: Angle<F>) -> Mat3<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat3::new( F::one(),  F::zero(), F::zero(),
                   F::zero(), c,         -s,
                   F::zero(), s,         c )
    }

    #[inline]
    pub fn from_angle_y(theta: Angle<F>) -> Mat3<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat3::new( c,         F::zero(), s,
                   F::zero(), F::one(),  F::zero(),
                   -s,        F::zero(), c )
    }

    #[inline]
    pub fn from_angle_z(theta: Angle<F>) -> Mat3<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat3::new(  c,        -s,         F::zero(),
                    s,         c,         F::zero(),
                    F::zero(), F::zero(), F::one() )
    }
}

impl Mat3<f32> {
    // https://en.wikipedia.org/w/index.php?title=Rotation_matrix
    //  #Rotation_matrix_from_axis_and_angle
    pub fn rotate_axis_angle(axis: Direction3<f32>, theta: Angle<f32>) -> Mat3<f32> {
        let axis: Vec3<f32> = From::from(axis);
        let x = &axis.x;
        let y = &axis.y;
        let z = &axis.z;
        let (s, c) = theta.as_radians().sin_cos();
        let ic = 1.0 - c;
        Mat3::new( x*x*ic + c   , x*y*ic - z*s ,  x*z*ic + y*s,
                   y*x*ic + z*s , y*y*ic + c   ,  y*z*ic - x*s,
                   z*x*ic - y*s , z*y*ic + x*s ,  z*z*ic + c   )
    }
}

impl Mat3<f64> {
    // https://en.wikipedia.org/w/index.php?title=Rotation_matrix
    //  #Rotation_matrix_from_axis_and_angle
    pub fn rotate_axis_angle(axis: Direction3<f64>, theta: Angle<f64>) -> Mat3<f64> {
        let axis: Vec3<f64> = From::from(axis);
        let x = &axis.x;
        let y = &axis.y;
        let z = &axis.z;
        let (s, c) = theta.as_radians().sin_cos();
        let ic = 1.0 - c;
        Mat3::new( x*x*ic + c   , x*y*ic - z*s ,  x*z*ic + y*s,
                   y*x*ic + z*s , y*y*ic + c   ,  y*z*ic - x*s,
                   z*x*ic - y*s , z*y*ic + x*s ,  z*z*ic + c   )
    }
}

impl<F: Float + One + Zero> Mat4<F> {
    #[inline]
    pub fn from_angle_x(theta: Angle<F>) -> Mat4<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat4::new( F::one(),  F::zero(), F::zero(), F::zero(),
                   F::zero(), c,        -s,         F::zero(),
                   F::zero(), s,         c,         F::zero(),
                   F::zero(), F::zero(), F::zero(), F::one() )
    }

    #[inline]
    pub fn from_angle_y(theta: Angle<F>) -> Mat4<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat4::new( c        , F::zero(),         s, F::zero(),
                   F::zero(), F::one(),  F::zero(), F::zero(),
                   -s       , F::zero(),         c, F::zero(),
                   F::zero(), F::zero(), F::zero(), F::one()   )
    }

    #[inline]
    pub fn from_angle_z(theta: Angle<F>) -> Mat4<F> {
        let (s, c) = theta.as_radians().sin_cos();
        Mat4::new( c        ,         s, F::zero(), F::zero(),
                   -s       ,         c, F::zero(), F::zero(),
                   F::zero(), F::zero(), F::one(),  F::zero(),
                   F::zero(), F::zero(), F::zero(), F::one()   )
    }
}

impl Mat4<f32> {
    // https://en.wikipedia.org/w/index.php?title=Rotation_matrix
    //  #Rotation_matrix_from_axis_and_angle
    pub fn rotate_axis_angle(axis: Direction3<f32>, theta: Angle<f32>) -> Mat4<f32> {
        let axis: Vec3<f32> = From::from(axis);
        let x = &axis.x;
        let y = &axis.y;
        let z = &axis.z;
        let (s, c) = theta.as_radians().sin_cos();
        let ic = 1.0 - c;
        Mat4::new( x*x*ic + c  , x*y*ic - z*s, x*z*ic + y*s,  0.0,
                   y*x*ic + z*s, y*y*ic + c  , y*z*ic - x*s,  0.0,
                   z*x*ic - y*s, z*y*ic + x*s, z*z*ic + c  ,  0.0,
                   0.0         , 0.0         , 0.0         ,  1.0 )
    }
}

impl Mat4<f64> {
    // https://en.wikipedia.org/w/index.php?title=Rotation_matrix
    //  #Rotation_matrix_from_axis_and_angle
    pub fn rotate_axis_angle(axis: Direction3<f64>, theta: Angle<f64>) -> Mat4<f64> {
        let axis: Vec3<f64> = From::from(axis);
        let x = &axis.x;
        let y = &axis.y;
        let z = &axis.z;
        let (s, c) = theta.as_radians().sin_cos();
        let ic = 1.0 - c;
        Mat4::new( x*x*ic + c  , x*y*ic - z*s, x*z*ic + y*s,  0.0,
                   y*x*ic + z*s, y*y*ic + c  , y*z*ic - x*s,  0.0,
                   z*x*ic - y*s, z*y*ic + x*s, z*z*ic + c  ,  0.0,
                   0.0         , 0.0         , 0.0         ,  1.0 )
    }
}

// -- Reflection --------------------------------------------------------------

impl Mat3<f32> {
    #[inline]
    /// Reflection matrix
    pub fn reflect_origin_plane(a: Direction3<f32>) -> Mat3<f32> {
        let a: Vec3<f32> = From::from(a);
        let x: f32 = a.x * -2.0;
        let y: f32 = a.y * -2.0;
        let z: f32 = a.z * -2.0;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Mat3::new( x * a.x + 1.0,   axay,           axaz,
                   axay,            y * a.y + 1.0,  ayaz,
                   axaz,            ayaz,           z * a.z + 1.0 )
    }
}

impl Mat3<f64> {
    #[inline]
    /// Reflection matrix
    pub fn reflect_origin_plane(a: Direction3<f64>) -> Mat3<f64> {
        let a: Vec3<f64> = From::from(a);
        let x: f64 = a.x * -2.0;
        let y: f64 = a.y * -2.0;
        let z: f64 = a.z * -2.0;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Mat3::new( x * a.x + 1.0,   axay,           axaz,
                   axay,            y * a.y + 1.0,  ayaz,
                   axaz,            ayaz,           z * a.z + 1.0 )
    }
}

impl Mat3<f32> {
    #[inline]
    /// Involution matrix
    pub fn involve_origin_plane(a: Direction3<f32>) -> Mat3<f32> {
        let a: Vec3<f32> = From::from(a);
        let x: f32 = a.x * 2.0;
        let y: f32 = a.y * 2.0;
        let z: f32 = a.z * 2.0;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Mat3::new( x * a.x - 1.0,   axay,           axaz,
                   axay,            y * a.y - 1.0,  ayaz,
                   axaz,            ayaz,           z * a.z - 1.0 )
    }
}

impl Mat3<f64> {
    #[inline]
    /// Involution matrix
    pub fn involve_origin_plane(a: Direction3<f64>) -> Mat3<f64> {
        let a: Vec3<f64> = From::from(a);
        let x: f64 = a.x * 2.0;
        let y: f64 = a.y * 2.0;
        let z: f64 = a.z * 2.0;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Mat3::new( x * a.x - 1.0,   axay,           axaz,
                   axay,            y * a.y - 1.0,  ayaz,
                   axaz,            ayaz,           z * a.z - 1.0 )
    }
}

// -- Scale -------------------------------------------------------------------

impl<F: Zero + One + Copy + Mul<F,Output=F>> Mat3<F> {
    #[inline]
    /// Scale matrix
    pub fn scale(a: &Vec3<F>) -> Mat3<F> {
        Mat3::new(a.x, F::zero(), F::zero(),
                  F::zero(), a.y, F::zero(),
                  F::zero(), F::zero(), a.z)
    }
}

impl<F: Zero + Copy> Mat4<F> {
    #[inline]
    /// Scale matrix
    pub fn scale(a: &Vec4<F>) -> Mat4<F> {
        Mat4::new(a.x, F::zero(), F::zero(), F::zero(),
                  F::zero(), a.y, F::zero(), F::zero(),
                  F::zero(), F::zero(), a.z, F::zero(),
                  F::zero(), F::zero(), F::zero(), a.w)
    }
}


impl Mat3<f32> {
    /// Scale along vector
    pub fn scale_in_direction(mut s: f32, a: Direction3<f32>) -> Mat3<f32> {
        let a: Vec3<f32> = From::from(a);
        s -= 1.0;
        let x = a.x * s;
        let y = a.y * s;
        let z = a.z * s;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Mat3::new(x * a.x + 1.0,  axay,           axaz,
                  axay,           y * a.y + 1.0,  ayaz,
                  axaz,           ayaz,           z * a.z + 1.0)
    }
}

impl Mat3<f64> {
    /// Scale along vector
    pub fn scale_in_direction(mut s: f64, a: Direction3<f64>) -> Mat3<f64> {
        let a: Vec3<f64> = From::from(a);
        s -= 1.0;
        let x = a.x * s;
        let y = a.y * s;
        let z = a.z * s;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Mat3::new(x * a.x + 1.0,  axay,           axaz,
                  axay,           y * a.y + 1.0,  ayaz,
                  axaz,           ayaz,           z * a.z + 1.0)
    }
}

impl Mat4<f32> {
    pub fn get_x_scale(&self) -> f32 {
        let scale = (self.x.x * self.x.x
                     + self.y.x * self.y.x
                     + self.z.x * self.z.x).sqrt();
        if self.x.x < 0.0 { -scale } else { scale }
    }

    pub fn get_y_scale(&self) -> f32 {
        let scale = (self.x.y * self.x.y
                     + self.y.y * self.y.y
                     + self.z.y * self.z.y).sqrt();
        if self.y.y < 0.0 { -scale } else { scale }
    }

    pub fn get_z_scale(&self) -> f32 {
        let scale = (self.x.z * self.x.z
                     + self.y.z * self.y.z
                     + self.z.z * self.z.z).sqrt();
        if self.z.z < 0.0 { -scale } else { scale }
    }
}

impl Mat4<f64> {
    pub fn get_x_scale(&self) -> f64 {
        let scale = (self.x.x * self.x.x
                     + self.y.x * self.y.x
                     + self.z.x * self.z.x).sqrt();
        if self.x.x < 0.0 { -scale } else { scale }
    }

    pub fn get_y_scale(&self) -> f64 {
        let scale = (self.x.y * self.x.y
                     + self.y.y * self.y.y
                     + self.z.y * self.z.y).sqrt();
        if self.y.y < 0.0 { -scale } else { scale }
    }

    pub fn get_z_scale(&self) -> f64 {
        let scale = (self.x.z * self.x.z
                     + self.y.z * self.y.z
                     + self.z.z * self.z.z).sqrt();
        if self.z.z < 0.0 { -scale } else { scale }
    }
}

// -- Skew --------------------------------------------------------------------

impl Mat3<f32> {
    /// Skew by give given angle in the given direction a, based on the projected
    /// length along the proj direction.  direction and proj MUST BE PERPENDICULAR
    /// or else results are undefined.
    pub fn skew(angle: f32, a: Direction3<f32>, proj: Direction3<f32>) -> Mat3<f32> {
        let a: Vec3<f32> = From::from(a);
        let b: Vec3<f32> = From::from(proj);
        let t = angle.tan();
        let x: f32 = a.x * t;
        let y: f32 = a.y * t;
        let z: f32 = a.z * t;
        Mat3::new(x * b.x + 1.0, x * b.y,       x * b.z,
                  y * b.x,       y * b.y + 1.0, y * b.z,
                  z * b.x,       z * b.y,       z * b.z + 1.0)
    }
}

impl Mat3<f64> {
    /// Skew by give given angle in the given direction a, based on the projected
    /// length along the proj direction.  direction and proj MUST BE PERPENDICULAR
    /// or else results are undefined.
    pub fn skew(angle: f64, a: Direction3<f64>, proj: Direction3<f64>) -> Mat3<f64> {
        let a: Vec3<f64> = From::from(a);
        let b: Vec3<f64> = From::from(proj);
        let t = angle.tan();
        let x: f64 = a.x * t;
        let y: f64 = a.y * t;
        let z: f64 = a.z * t;
        Mat3::new(x * b.x + 1.0, x * b.y,       x * b.z,
                  y * b.x,       y * b.y + 1.0, y * b.z,
                  z * b.x,       z * b.y,       z * b.z + 1.0)
    }
}

// ----------------------------------------------------------------------------
// Convert between f32 and f64

impl From<Mat3<f32>> for Mat3<f64> {
    fn from(m: Mat3<f32>) -> Mat3<f64> {
        Mat3 {
            x: Vec3 { x: m.x.x as f64, y: m.x.y as f64, z: m.x.z as f64 },
            y: Vec3 { x: m.y.x as f64, y: m.y.y as f64, z: m.y.z as f64 },
            z: Vec3 { x: m.z.x as f64, y: m.z.y as f64, z: m.z.z as f64 },
        }
    }
}

impl From<Mat3<f64>> for Mat3<f32> {
    fn from(m: Mat3<f64>) -> Mat3<f32> {
        Mat3 {
            x: Vec3 { x: m.x.x as f32, y: m.x.y as f32, z: m.x.z as f32 },
            y: Vec3 { x: m.y.x as f32, y: m.y.y as f32, z: m.y.z as f32 },
            z: Vec3 { x: m.z.x as f32, y: m.z.y as f32, z: m.z.z as f32 },
        }
    }
}

impl From<Mat4<f32>> for Mat4<f64> {
    fn from(m: Mat4<f32>) -> Mat4<f64> {
        Mat4 {
            x: Vec4 { x: m.x.x as f64, y: m.x.y as f64, z: m.x.z as f64, w: m.x.w as f64 },
            y: Vec4 { x: m.y.x as f64, y: m.y.y as f64, z: m.y.z as f64, w: m.y.w as f64 },
            z: Vec4 { x: m.z.x as f64, y: m.z.y as f64, z: m.z.z as f64, w: m.z.w as f64 },
            p: Vec4 { x: m.p.x as f64, y: m.p.y as f64, z: m.p.z as f64, w: m.p.w as f64 },
        }
    }
}

impl From<Mat4<f64>> for Mat4<f32> {
    fn from(m: Mat4<f64>) -> Mat4<f32> {
        Mat4 {
            x: Vec4 { x: m.x.x as f32, y: m.x.y as f32, z: m.x.z as f32, w: m.x.w as f32 },
            y: Vec4 { x: m.y.x as f32, y: m.y.y as f32, z: m.y.z as f32, w: m.y.w as f32 },
            z: Vec4 { x: m.z.x as f32, y: m.z.y as f32, z: m.z.z as f32, w: m.z.w as f32 },
            p: Vec4 { x: m.p.x as f32, y: m.p.y as f32, z: m.p.z as f32, w: m.p.w as f32 },
        }
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{Mat2, Mat3, Mat4, Angle};
    use super::super::vector::{Vec2, Vec3, Vec4, Direction3};

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

    #[test]
    fn test_axis_angle() {
        let axis: Direction3<f32> = From::from(Vec3::new(1.0, 0.0, 0.0));
        let angle = Angle::new_radians(::std::f32::consts::FRAC_PI_4);;

        let start: Mat4<f32> = Mat4::new(
            1.0, 0.0, 0.0, 5.0,
            0.0, 1.0, 0.0, 5.0,
            0.0, 0.0, 1.0, 5.0,
            0.0, 0.0, 0.0, 1.0 );

        let rot = Mat4::<f32>::rotate_axis_angle(axis, angle);

        let end = &rot * &start;

        let (s, c) = angle.as_radians().sin_cos();
        // This equality comparison works even with floating point inaccuracies.
        // But ideally we need ULPS comparison functions for vectors and matrices.
        assert_eq!(end, Mat4::new(
            1.0, 0.0, 0.0, 5.0,
            0.0, c,   -s,  0.0,
            0.0, s,   c,   c*10.0,
            0.0, 0.0, 0.0, 1.0));
    }
}
