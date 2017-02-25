
use std::default::Default;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
// FIXME: once simd is part of std and stable, use it
//use simd::maths::sqrt::Rsqrt;

#[derive(Debug, Clone, Copy)]
pub struct Vec4(pub f32, pub f32, pub f32, pub f32);

impl Default for Vec4 {
    fn default() -> Vec4 {
        Vec4(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32)
    }
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4(x,y,z,w)
    }

    #[inline]
    pub fn zero() -> Vec4 {
        Vec4(0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn squared_magnitude(&self) -> f32 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2 + self.3 * self.3
    }

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
        self.0 /= mag;
        self.1 /= mag;
        self.2 /= mag;
        self.3 /= mag;
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;

    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            3 => &self.3,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            3 => &mut self.3,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: f32) -> Vec4 {
        Vec4(self.0 * rhs, self.1 * rhs, self.2 * rhs, self.3 * rhs)
    }
}

impl MulAssign<f32> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
        self.3 *= rhs;
    }
}

impl Div<f32> for Vec4 {
    type Output = Vec4;

    #[inline]
    fn div(self, rhs: f32) -> Vec4 {
        Vec4(self.0 / rhs, self.1 / rhs, self.2 / rhs, self.3 / rhs)
    }
}

impl DivAssign<f32> for Vec4 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.0 /= rhs;
        self.1 /= rhs;
        self.2 /= rhs;
        self.3 /= rhs;
    }
}

impl Neg for Vec4 {
    type Output = Vec4;

    #[inline]
    fn neg(self) -> Vec4 {
        Vec4(-self.0, -self.1, -self.2, -self.3)
    }
}

impl Add<Vec4> for Vec4 {
    type Output = Vec4;

    #[inline]
    fn add(self, other: Vec4) -> Vec4 {
        Vec4(self.0 + other.0, self.1 + other.1, self.2 + other.2, self.3 + other.3)
    }
}

impl<'a> AddAssign<&'a Vec4> for Vec4 {
    #[inline]
    fn add_assign(&mut self, other: &'a Vec4) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
        self.3 += other.3;
    }
}

impl Sub<Vec4> for Vec4 {
    type Output = Vec4;

    #[inline]
    fn sub(self, other: Vec4) -> Vec4 {
        Vec4(self.0 - other.0, self.1 - other.1, self.2 - other.2, self.3 - other.3)
    }
}

impl<'a> SubAssign<&'a Vec4> for Vec4 {
    #[inline]
    fn sub_assign(&mut self, other: &'a Vec4) {
        self.0 -= other.0;
        self.1 -= other.1;
        self.2 -= other.2;
        self.3 -= other.3;
    }
}
