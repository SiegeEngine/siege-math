
use std::ops::{Add, Sub, Mul,
               AddAssign, SubAssign, MulAssign,
               Neg};
use std::default::Default;
use Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Quat<F> {
    pub v: Vec3<F>,
    pub w: F
}

impl<F: Default> Default for Quat<F> {
    fn default() -> Quat<F> {
        Quat {
            v: Default::default(),
            w: Default::default()
        }
    }
}

impl<F: Add<Output=F>>
    Add for Quat<F>
{
    type Output = Quat<F>;

    fn add(self, rhs: Quat<F>) -> Quat<F> {
        Quat {
            v: self.v + rhs.v,
            w: self.w + rhs.w,
        }
    }
}

impl<F: Add<Output=F> + AddAssign<F> + Copy>
    AddAssign for Quat<F>
{
    fn add_assign(&mut self, rhs: Quat<F>) {
        self.v += rhs.v;
        self.w += rhs.w;
    }
}

impl<F: Sub<Output=F>>
    Sub for Quat<F>
{
    type Output = Quat<F>;

    fn sub(self, rhs: Quat<F>) -> Quat<F> {
        Quat {
            v: self.v - rhs.v,
            w: self.w - rhs.w,
        }
    }
}

impl<F: Sub<Output=F> + SubAssign<F> + Copy>
    SubAssign for Quat<F>
{
    fn sub_assign(&mut self, rhs: Quat<F>) {
        self.v -= rhs.v;
        self.w -= rhs.w;
    }
}

impl<F: Copy + Mul<F,Output=F>>
    Mul<F> for Quat<F>
{
    type Output = Quat<F>;

    fn mul(self, rhs: F) -> Quat<F> {
        Quat {
            v: self.v * rhs,
            w: self.w * rhs,
        }
    }
}

impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F>>
    Mul for Quat<F>
{
    type Output = Quat<F>;

    fn mul(self, rhs: Quat<F>) -> Quat<F> {
        Quat {
            v: self.v.cross(rhs.v)  +  rhs.v * self.w  +  self.v * rhs.w,
            w: self.w * rhs.w  -  self.v.dot(rhs.v)
        }
    }
}

impl<F: Copy + Mul<F,Output=F> + MulAssign<F>>
    MulAssign<F> for Quat<F>
{
    fn mul_assign(&mut self, rhs: F) {
        self.v *= rhs;
        self.w *= rhs;
    }
}

impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F>>
    MulAssign for Quat<F>
{
    fn mul_assign(&mut self, rhs: Quat<F>) {
        self.v = self.v.cross(rhs.v)  +  rhs.v * self.w  +  self.v * rhs.w;
        self.w = self.w * rhs.w  -  self.v.dot(rhs.v);
    }
}



impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F> + Neg<Output=F>>
    Quat<F>
{
    pub fn conjugate(&self) -> Quat<F> {
        Quat {
            v: -self.v,
            w: self.w
        }
    }

    pub fn squared_magnitude(&self) -> Quat<F> {
        *self * self.conjugate()
    }
}
