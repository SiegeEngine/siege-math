
use std::default::Default;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
// FIXME: once simd is part of std and stable, use it
//use simd::maths::sqrt::Rsqrt;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vec3(pub f32, pub f32, pub f32);

impl Default for Vec3 {
    fn default() -> Vec3 {
        Vec3(0.0_f32, 0.0_f32, 0.0_f32)
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3(x,y,z)
    }

    #[inline]
    pub fn zero() -> Vec3 {
        Vec3(0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn squared_magnitude(&self) -> f32 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
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
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn div(self, rhs: f32) -> Vec3 {
        Vec3(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.0 /= rhs;
        self.1 /= rhs;
        self.2 /= rhs;
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    #[inline]
    fn neg(self) -> Vec3 {
        Vec3(-self.0, -self.1, -self.2)
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn add(self, other: Vec3) -> Vec3 {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl<'a> AddAssign<&'a Vec3> for Vec3 {
    #[inline]
    fn add_assign(&mut self, other: &'a Vec3) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl<'a> SubAssign<&'a Vec3> for Vec3 {
    #[inline]
    fn sub_assign(&mut self, other: &'a Vec3) {
        self.0 -= other.0;
        self.1 -= other.1;
        self.2 -= other.2;
    }
}
