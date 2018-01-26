
use std::ops::{Add, Sub, Mul, Div,
               AddAssign, SubAssign, MulAssign,
               Neg};
use num_traits::{Zero, One, Float};
use std::default::Default;
use float_cmp::{Ulps, ApproxEqUlps};
use {Vec3, Mat3};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Quat<F> {
    pub v: Vec3<F>,
    pub w: F
}

impl<F: Copy + Float + Mul<F,Output=F> + Add<F,Output=F> + Div<F,Output=F>> Quat<F> {
    pub fn new(v: Vec3<F>, w: F) -> Quat<F> {
        Quat {
            v: v,
            w: w,
        }
    }
}

impl<F: Zero + One> Quat<F> {
    pub fn identity() -> Quat<F> {
        Quat {
            v: Vec3::new(F::zero(), F::zero(), F::zero()),
            w: F::one()
        }
    }
}

impl<F: Zero + One> Default for Quat<F> {
    fn default() -> Quat<F> {
        Quat::identity()
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

impl From<Quat<f64>> for Quat<f32> {
    fn from(q: Quat<f64>) -> Quat<f32> {
        Quat {
            v: From::from(q.v),
            w: q.w as f32
        }
    }
}

impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F> + Neg<Output=F>>
    Quat<F>
{
    pub fn squared_magnitude(&self) -> F {
        self.w * self.w + self.v.squared_magnitude()
    }
}

impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F> + Neg<Output=F> + Float>
    Quat<F>
{
    pub fn magnitude(&self) -> F {
        self.squared_magnitude().sqrt()
    }
}

impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F> + Neg<Output=F> + Float>
    Quat<F>
{
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        self.v = self.v / mag;
        self.w = self.w / mag;
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

impl<F: Copy + AddAssign<F>>
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

impl<F: Copy + Add<F,Output=F> + Mul<F,Output=F>>
    Quat<F>
{
    pub fn dot(&self, other: Quat<F>) -> F
    {
        (self.w * other.w) + self.v.dot(other.v)
    }
}

// Hamiltonian product
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


impl<F: Copy + Neg<Output=F>>
    Quat<F>
{
    pub fn conjugate(&self) -> Quat<F> {
        Quat {
            v: -self.v,
            w: self.w
        }
    }
}

impl<F: Copy + Add<Output=F> + Mul<Output=F> + Sub<Output=F> + Neg<Output=F>>
    Quat<F>
{
    pub fn rotate(&self, v: Vec3<F>) -> Vec3<F> {
        let dot = v.dot(self.v);

        v * ((self.w * self.w) - (self.v.x * self.v.x) - (self.v.y * self.v.y) - (self.v.z * self.v.z))
            + self.v * (dot + dot)
            + self.v.cross(v) * (self.w + self.w)
    }
}

impl From<Quat<f32>> for Mat3<f32> {
    fn from(mut q: Quat<f32>) -> Mat3<f32> {
        q.normalize();

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
    fn from(mut q: Quat<f64>) -> Mat3<f64> {
        q.normalize();

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
            let f = 0.25 / w;
            x = (m.z.y - m.y.z) * f;
            y = (m.x.z - m.z.x) * f;
            z = (m.y.x - m.x.y) * f;
        }
        else if (m.x.x > m.y.y) && (m.x.x > m.z.z) {
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
            z = (m.z.y + m.y.z) * f;
            w = (m.x.z - m.z.x) * f;
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

impl<F: Ulps + ApproxEqUlps<Flt=F>> ApproxEqUlps for Quat<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.v.approx_eq_ulps(&other.v, ulps) &&
            self.w.approx_eq_ulps(&other.w, ulps)
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
            let f = 0.25 / w;
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
            z = (m.z.y + m.y.z) * f;
            w = (m.x.z - m.z.x) * f;
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

#[cfg(test)]
mod tests {
    use {Quat, Vec3, Mat3, Direction3, Angle};

    #[test]
    fn test_quat_basic() {
        let q = Quat::<f64>::new(
            Vec3::<f64>::new(3.0, 4.0, 5.0),
            6.0);
        assert_eq!(q.v.x, 3.0);
        assert_eq!(q.v.y, 4.0);
        assert_eq!(q.v.z, 5.0);
        assert_eq!(q.w, 6.0);

        let q = Quat::<f64>::identity();
        assert_eq!(q.v.x, 0.0);
        assert_eq!(q.v.y, 0.0);
        assert_eq!(q.v.z, 0.0);
        assert_eq!(q.w, 1.0);

        let q2 = Quat::<f64>::default();
        assert_eq!(q, q2);

        let mut q = Quat::<f64>::new(
            Vec3::<f64>::new(3.0, 4.0, 5.0),
            6.0);
        q.normalize();
        assert!(0.999999999 < q.magnitude());
        assert!(q.magnitude() < 1.000000001);
    }

    #[test]
    fn test_quat_mat_conversion() {
        let v = Vec3::<f32>::new(1.0, 0.2, -0.3);
        let mut q = Quat::<f32>::new(v, 5.0);
        q.normalize();

        let m: Mat3<f32> = From::from(q);
        println!("m: {:?}", m);

        // Conversion through a matrix could return the
        // conjugate (which represents the same rotation)
        // so we have to check for either
        let q2: Quat<f32> = From::from(m);
        let q2c: Quat<f32> = q2.conjugate();

        // FIXME...
        let xdiff = q.v.x - q2.v.x;
        let ydiff = q.v.y - q2.v.y;
        let zdiff = q.v.z - q2.v.z;
        let wdiff = q.w - q2.w;
        assert!(-0.000001 < xdiff && xdiff < 0.000001);
        assert!(-0.000001 < ydiff && ydiff < 0.000001);
        assert!(-0.000001 < zdiff && zdiff < 0.000001);
        assert!(-0.000001 < wdiff && wdiff < 0.000001);
    }

    /*
    #[test]
    fn test_quat_2() {
        let axis: Direction3<f32> = From::from(Vec3::new(1.0, 0.0, 0.0));
        let angle = Angle::new_radians(1.0);
        let mr = Mat3::<f32>::rotate_axis_angle(axis, angle);
        let q: Quat<f32> = From::from(mr);

        let v: Vec3<f32> = Vec3::new(5.0, 23.0, -3.5);
        let vr1 = q.rotate(v);

        // Compare to another method of rotation
        let vq = Quat::<f32> { v: v, w: 0.0 };
        let qc = q.conjugate();
        let vr2 = (q * vq * qc).v;

        println!("vr1={:?} vr2={:?}", vr1, vr2);
    }
*/
}
