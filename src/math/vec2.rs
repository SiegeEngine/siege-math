
use std::default::Default;
use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Neg,
               Add, AddAssign, Sub, SubAssign};
// FIXME: once simd is part of std and stable, use it
//use simd::maths::sqrt::Rsqrt;

#[derive(Debug, Clone, Copy)]
pub struct Vec2(pub f32, pub f32);

impl Default for Vec2 {
    fn default() -> Vec2 {
        Vec2(0.0_f32, 0.0_f32)
    }
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2(x,y)
    }

    #[inline]
    pub fn zero() -> Vec2 {
        Vec2(0.0, 0.0)
    }

    #[inline]
    pub fn squared_magnitude(&self) -> f32 {
        self.0 * self.0 + self.1 * self.1
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
    }
}

impl Index<usize> for Vec2 {
    type Output = f32;

    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl IndexMut<usize> for Vec2 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.0,
            1 => &mut self.1,
            _ => panic!("Index out of bounds for Vec2"),
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: f32) -> Vec2 {
        Vec2(self.0 * rhs, self.1 * rhs)
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl Div<f32> for Vec2 {
    type Output = Vec2;

    #[inline]
    fn div(self, rhs: f32) -> Vec2 {
        Vec2(self.0 / rhs, self.1 / rhs)
    }
}

impl DivAssign<f32> for Vec2 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.0 /= rhs;
        self.1 /= rhs;
    }
}

impl Neg for Vec2 {
    type Output = Vec2;

    #[inline]
    fn neg(self) -> Vec2 {
        Vec2(-self.0, -self.1)
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Vec2;

    #[inline]
    fn add(self, other: Vec2) -> Vec2 {
        Vec2(self.0 + other.0, self.1 + other.1)
    }
}

impl<'a> AddAssign<&'a Vec2> for Vec2 {
    #[inline]
    fn add_assign(&mut self, other: &'a Vec2) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Vec2;

    #[inline]
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2(self.0 - other.0, self.1 - other.1)
    }
}

impl<'a> SubAssign<&'a Vec2> for Vec2 {
    #[inline]
    fn sub_assign(&mut self, other: &'a Vec2) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}
