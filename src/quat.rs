
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
use num_traits::{Zero, One, Float};
use std::default::Default;
use float_cmp::{Ulps, ApproxEqUlps};
use {Vec3, Mat3, Angle, Direction3};

// Quaternion (general)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Quat<F> {
    pub v: Vec3<F>,
    pub w: F
}

/// Normalized unit quaternion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct NQuat<F> {
    v: Vec3<F>,
    w: F
}

impl<F> Quat<F> {
    pub fn new(v: Vec3<F>, w: F) -> Quat<F> {
        Quat {
            v: v,
            w: w,
        }
    }
}

impl<F> NQuat<F> {
    pub fn new_isnormal(v: Vec3<F>, w: F) -> NQuat<F> {
        NQuat {
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
impl<F: Zero + One> NQuat<F> {
    pub fn identity() -> NQuat<F> {
        NQuat {
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
impl<F: Zero + One> Default for NQuat<F> {
    fn default() -> NQuat<F> {
        NQuat::identity()
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

impl From<NQuat<f32>> for NQuat<f64> {
    fn from(q: NQuat<f32>) -> NQuat<f64> {
        NQuat {
            v: From::from(q.v),
            w: q.w as f64
        }
    }
}

impl From<NQuat<f64>> for NQuat<f32> {
    fn from(q: NQuat<f64>) -> NQuat<f32> {
        NQuat {
            v: From::from(q.v),
            w: q.w as f32
        }
    }
}


impl<F: Copy + Add<Output=F> + Sub<Output=F> + Mul<Output=F> + Neg<Output=F> + Float>
    From<Quat<F>> for NQuat<F>
{
    fn from(q: Quat<F>) -> NQuat<F> {
        let mag = q.magnitude();
        NQuat {
            v: q.v / mag,
            w: q.w / mag
        }
    }
}

impl<F> From<NQuat<F>> for Quat<F> {
    fn from(nq: NQuat<F>) -> Quat<F> {
        Quat { v: nq.v, w: nq.w }
    }
}

impl<F: Zero + One + Float + Mul<F,Output=F>> NQuat<F> {
    pub fn from_axis_angle(axis: &Direction3<F>, angle: &Angle<F>) -> NQuat<F>
    {
        let two = F::one() + F::one();
        let (s,c) = (angle.as_radians() / two).sin_cos();
        let q = NQuat {
            v: Vec3::new(axis.x * s, axis.y * s, axis.z * s),
            w: c
        };
        From::from(q)
    }
}

impl<F: Zero + One + Float + Mul<F,Output=F> + DivAssign> NQuat<F> {
    pub fn as_axis_angle(&self) -> (Direction3<F>, Angle<F>)
    {
        let two = F::one() + F::one();
        let angle = self.w.acos() * two;
        let axis = self.v;
        (From::from(axis), Angle::from_radians(angle))
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

impl<F: Copy + Add<F,Output=F> + Mul<F,Output=F>>
    NQuat<F>
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

impl<F: Copy + Float + Add<Output=F> + Sub<Output=F> + Mul<Output=F>>
    Mul for NQuat<F>
{
    type Output = NQuat<F>;

    fn mul(self, rhs: NQuat<F>) -> NQuat<F> {
        let a: Quat<F> = From::from(self);
        let b: Quat<F> = From::from(rhs);
        From::from(a * b)
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

impl<F: Copy + Neg<Output=F>>
    NQuat<F>
{
    pub fn conjugate(&self) -> NQuat<F> {
        NQuat {
            v: -self.v,
            w: self.w
        }
    }
}

impl<F: Copy + Add<Output=F> + Mul<Output=F> + Sub<Output=F> + Neg<Output=F>>
    NQuat<F>
{
    pub fn rotate(&self, v: Vec3<F>) -> Vec3<F> {
        let dot = v.dot(self.v);

        v * ((self.w * self.w) - (self.v.x * self.v.x) - (self.v.y * self.v.y) - (self.v.z * self.v.z))
            + self.v * (dot + dot)
            + self.v.cross(v) * (self.w + self.w)
    }
}

impl From<NQuat<f32>> for Mat3<f32> {
    fn from(q: NQuat<f32>) -> Mat3<f32> {
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

impl From<NQuat<f64>> for Mat3<f64> {
    fn from(q: NQuat<f64>) -> Mat3<f64> {
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

impl From<Mat3<f32>> for NQuat<f32> {
    fn from(m: Mat3<f32>) -> NQuat<f32> {
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

        NQuat { v: Vec3 { x: x, y: y, z: z }, w: w }
    }
}

impl From<Mat3<f64>> for NQuat<f64> {
    fn from(m: Mat3<f64>) -> NQuat<f64> {
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

        NQuat { v: Vec3 { x: x, y: y, z: z }, w: w }
    }
}

impl<F: Ulps + ApproxEqUlps<Flt=F>> ApproxEqUlps for Quat<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.v.approx_eq_ulps(&other.v, ulps) &&
            self.w.approx_eq_ulps(&other.w, ulps)
    }
}

impl<F: Ulps + ApproxEqUlps<Flt=F>> ApproxEqUlps for NQuat<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.v.approx_eq_ulps(&other.v, ulps) &&
            self.w.approx_eq_ulps(&other.w, ulps)
    }
}

#[cfg(test)]
mod tests {
    use {Quat, NQuat, Vec3, Mat3, Angle, Direction3};

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

        let q: NQuat<f64> = From::from(
            Quat::<f64>::new(Vec3::<f64>::new(3.0, 4.0, 5.0), 6.0)
        );
        let q2: Quat<f64> = From::from(q);
        assert!(0.999999999 < q2.magnitude());
        assert!(q2.magnitude() < 1.000000001);
    }

    #[test]
    fn test_quat_mat_conversion() {
        use float_cmp::ApproxEqUlps;

        let v = Vec3::<f32>::new(1.0, 0.2, -0.3);
        let q: NQuat<f32> = From::from(
            Quat::<f32>::new(v, 5.0));

        let m: Mat3<f32> = From::from(q);

        // Conversion through a matrix could return the
        // conjugate (which represents the same rotation)
        // so we have to check for either
        let q2: NQuat<f32> = From::from(m);
        let q2c: NQuat<f32> = q2.conjugate();

        assert!(q2.approx_eq_ulps(&q, 2) ||
                q2c.approx_eq_ulps(&q, 2));
    }

    #[test]
    fn test_axis_angle() {
        use float_cmp::ApproxEqUlps;

        let axis: Direction3<f32> = From::from(Vec3::<f32>::new(1.0, 1.0, 1.0));
        let angle = Angle::<f32>::from_degrees(90.0);
        let q = NQuat::<f32>::from_axis_angle(&axis, &angle);
        let (axis2, angle2) = q.as_axis_angle();
        println!("axis {:?} angle {:?} axis {:?} angle {:?}",
                 axis, angle, axis2, angle2);
        assert!(axis.approx_eq_ulps(&axis2, 2));
        assert!(angle.approx_eq_ulps(&angle2, 2));
    }

    #[test]
    fn test_rotation_x() {
        use float_cmp::ApproxEqUlps;

        let axis: Direction3<f32> = From::from(Vec3::<f32>::new(1.0, 0.0, 0.0));
        let angle = Angle::<f32>::from_degrees(90.0);
        let q = NQuat::<f32>::from_axis_angle(&axis, &angle);

        let object = Vec3::<f32>::new(10.0, 5.0, 3.0);

        let object2 = q.rotate(object);

        assert!(object2.x.approx_eq_ulps(&10.0, 2));
        assert!(object2.y.approx_eq_ulps(&-3.0, 2));
        assert!(object2.z.approx_eq_ulps(&5.0, 2));
    }
}
