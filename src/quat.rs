
use std::ops::{Add, Sub, Mul,
               AddAssign, SubAssign, MulAssign,
               Neg};
use std::default::Default;
use {Vec3, Mat3};

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

impl From<Quat<f64>> for Quat<f32> {
    fn from(q: Quat<f64>) -> Quat<f32> {
        Quat {
            v: From::from(q.v),
            w: q.w as f32
        }
    }
}

impl From<Quat<f32>> for Quat<f64> {
    fn from(q: Quat<f32>) -> Quat<f64> {
        Quat {
            v: From::from(q.v),
            w: q.w as f64
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

impl Quat<f32> {
    pub fn rotate(&self, v: Vec3<f32>) -> Vec3<f32> {
        v * ((self.w * self.w) - (self.v.x * self.v.x) - (self.v.y * self.v.y) - (self.v.z * self.v.z))
            + self.v * (v.dot(self.v) * 2.0)
            + self.v.cross(v) * (self.w * 2.0)
    }
}

impl Quat<f64> {
    pub fn rotate(&self, v: Vec3<f64>) -> Vec3<f64> {
        v * ((self.w * self.w) - (self.v.x * self.v.x) - (self.v.y * self.v.y) - (self.v.z * self.v.z))
            + self.v * (v.dot(self.v) * 2.0)
            + self.v.cross(v) * (self.w * 2.0)
    }
}

impl From<Quat<f32>> for Mat3<f32> {
    fn from(q: Quat<f32>) -> Mat3<f32> {
        let x2 = q.v.x * q.v.x;
        let y2 = q.v.y * q.v.y;
        let z2 = q.v.z * q.v.z;
        let xy = q.v.x * q.v.y;
        let xz = q.v.x * q.v.z;
        let yz = q.v.y * q.v.z;
        let wx = q.w * q.v.x;
        let wy = q.w * q.v.y;
        let wz = q.w * q.v.z;

        Mat3::new(
            1.0 - 2.0 * (y2 + z2),        2.0 * (xy - wz),       2.0 * (xz + wy),
            2.0       * (xy + wz),  1.0 - 2.0 * (x2 + z2),       2.0 * (yz - wx),
            2.0       * (xz - wy),        2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)
        )
    }
}

impl From<Quat<f64>> for Mat3<f64> {
    fn from(q: Quat<f64>) -> Mat3<f64> {
        let x2 = q.v.x * q.v.x;
        let y2 = q.v.y * q.v.y;
        let z2 = q.v.z * q.v.z;
        let xy = q.v.x * q.v.y;
        let xz = q.v.x * q.v.z;
        let yz = q.v.y * q.v.z;
        let wx = q.w * q.v.x;
        let wy = q.w * q.v.y;
        let wz = q.w * q.v.z;

        Mat3::new(
            1.0 - 2.0 * (y2 + z2),        2.0 * (xy - wz),       2.0 * (xz + wy),
            2.0       * (xy + wz),  1.0 - 2.0 * (x2 + z2),       2.0 * (yz - wx),
            2.0       * (xz - wy),        2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)
        )
    }
}

impl From<Mat3<f32>> for Quat<f32> {
    fn from(m: Mat3<f32>) -> Quat<f32> {
        let sum = m.x.x + m.y.y + m.z.z;
        let x;
        let y;
        let z;
        let w;
        if sum>0.0 {
            w = (sum + 1.0).sqrt() * 0.5;
            let f = 0.25 / 2.0;
            x = (m.z.y - m.y.z) * f;
            y = (m.x.z - m.z.x) * f;
            z = (m.y.x - m.x.y) * f;
        }
        else if m.x.x > m.y.y && m.x.x > m.z.z {
            x = (m.x.x - m.y.y - m.z.z + 1.0).sqrt() * 0.5;
            let f = 0.25 / x;
            y = (m.y.x + m.x.y) * f;
            z = (m.x.z + m.z.x) * f;
            w = (m.z.y - m.y.z) * f;

        }
        else if m.y.y > m.z.z {
            y = (m.y.y - m.x.x - m.z.z + 1.0).sqrt() * 0.5;
            let f = 0.25 / y;
            x = (m.y.x + m.x.y) * f;
            z = (m.x.z + m.z.x) * f;
            w = (m.z.y - m.y.z) * f;
        }
        else {
            z = (m.z.z - m.x.x - m.y.y + 1.0).sqrt() * 0.5;
            let f = 0.25 / z;
            x = (m.x.z + m.z.x) * f;
            y = (m.z.y + m.y.z) * f;
            w = (m.y.x - m.x.y) * f;
        }

        Quat { v: Vec3 { x: x, y: y, z: z }, w: w }
    }
}

impl From<Mat3<f64>> for Quat<f64> {
    fn from(m: Mat3<f64>) -> Quat<f64> {
        let sum = m.x.x + m.y.y + m.z.z;
        let x;
        let y;
        let z;
        let w;
        if sum>0.0 {
            w = (sum + 1.0).sqrt() * 0.5;
            let f = 0.25 / 2.0;
            x = (m.z.y - m.y.z) * f;
            y = (m.x.z - m.z.x) * f;
            z = (m.y.x - m.x.y) * f;
        }
        else if m.x.x > m.y.y && m.x.x > m.z.z {
            x = (m.x.x - m.y.y - m.z.z + 1.0).sqrt() * 0.5;
            let f = 0.25 / x;
            y = (m.y.x + m.x.y) * f;
            z = (m.x.z + m.z.x) * f;
            w = (m.z.y - m.y.z) * f;

        }
        else if m.y.y > m.z.z {
            y = (m.y.y - m.x.x - m.z.z + 1.0).sqrt() * 0.5;
            let f = 0.25 / y;
            x = (m.y.x + m.x.y) * f;
            z = (m.x.z + m.z.x) * f;
            w = (m.z.y - m.y.z) * f;
        }
        else {
            z = (m.z.z - m.x.x - m.y.y + 1.0).sqrt() * 0.5;
            let f = 0.25 / z;
            x = (m.x.z + m.z.x) * f;
            y = (m.z.y + m.y.z) * f;
            w = (m.y.x - m.x.y) * f;
        }

        Quat { v: Vec3 { x: x, y: y, z: z }, w: w }
    }
}
