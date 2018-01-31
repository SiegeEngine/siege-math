
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
use num_traits::NumCast;
use std::default::Default;
use float_cmp::{Ulps, ApproxEqUlps};
use {FullFloat, Vec3, Mat3, Angle, Direction3};

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

impl<F: FullFloat> Quat<F> {
    pub fn new(v: Vec3<F>, w: F) -> Quat<F> {
        Quat {
            v: v,
            w: w,
        }
    }
}

impl<F: FullFloat> NQuat<F> {
    pub fn new_isnormal(v: Vec3<F>, w: F) -> NQuat<F> {
        let q = NQuat {
            v: v,
            w: w,
        };

        assert!((w*w + v.squared_magnitude()).sqrt().approx_eq_ulps(
            &F::one(),
            <<F as ApproxEqUlps>::Flt as Ulps>::default_ulps()
        ));

        q
    }
}

impl<F: FullFloat> Quat<F> {
    pub fn identity() -> Quat<F> {
        Quat {
            v: Vec3::new(F::zero(), F::zero(), F::zero()),
            w: F::one()
        }
    }
}
impl<F: FullFloat> NQuat<F> {
    pub fn identity() -> NQuat<F> {
        NQuat::new_isnormal(
            Vec3::new(F::zero(), F::zero(), F::zero()),
            F::one())
    }
}

impl<F: FullFloat> Default for Quat<F> {
    fn default() -> Quat<F> {
        Quat::identity()
    }
}

impl<F: FullFloat> Default for NQuat<F> {
    fn default() -> NQuat<F> {
        NQuat::identity()
    }
}

// ----------------------------------------------------------------------------
// casting between float types

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
        NQuat::new_isnormal(
            From::from(q.v),
            q.w as f64)
    }
}

impl From<NQuat<f64>> for NQuat<f32> {
    fn from(q: NQuat<f64>) -> NQuat<f32> {
        NQuat::new_isnormal(
            From::from(q.v),
            q.w as f32)
    }
}

// ----------------------------------------------------------------------------
// Casting to/from normal form

impl<F: FullFloat> From<Quat<F>> for NQuat<F>
{
    fn from(q: Quat<F>) -> NQuat<F> {
        let mag = q.magnitude();
        NQuat::new_isnormal(
            q.v / mag,
            q.w / mag)
    }
}

impl<F: FullFloat> From<NQuat<F>> for Quat<F> {
    fn from(nq: NQuat<F>) -> Quat<F> {
        Quat { v: nq.v, w: nq.w }
    }
}

// ----------------------------------------------------------------------------
// Axis/Angle

impl<F: FullFloat> NQuat<F> {
    // This always yields normal quats (tested)
    pub fn from_axis_angle(axis: &Direction3<F>, angle: &Angle<F>) -> NQuat<F>
    {
        let two: F = NumCast::from(2.0_f32).unwrap();
        let (s,c) = (angle.as_radians() / two).sin_cos();
        let q = Quat {
            v: Vec3::new(axis.x * s, axis.y * s, axis.z * s),
            w: c
        };
        From::from(q)
    }
}

impl<F: FullFloat> NQuat<F> {
    pub fn as_axis_angle(&self) -> (Direction3<F>, Angle<F>)
    {
        let two: F = NumCast::from(2.0_f32).unwrap();
        let angle = self.w.acos() * two;
        let axis = self.v;
        (From::from(axis), Angle::from_radians(angle))
    }
}

// ----------------------------------------------------------------------------
// Compute the quat that rotates from start to end

impl<F: FullFloat> NQuat<F> {
    // This returns None if start/end are the same or opposite)
    pub fn from_directions(start: Direction3<F>, end: Direction3<F>) -> Option<NQuat<F>>
    {
        let two: F = NumCast::from(2.0_f32).unwrap();
        let e = start.dot(end);
        if e==-F::one() {
            return None;
        }
        let term = (two * (F::one() + e)).sqrt();
        let v: Vec3<F> = start.cross(end);
        Some(NQuat::new_isnormal(
            v / term,
            term / two))
    }
}

// ----------------------------------------------------------------------------
// Magnitude

impl<F: FullFloat> Quat<F>
{
    pub fn squared_magnitude(&self) -> F {
        self.w * self.w + self.v.squared_magnitude()
    }
}

impl<F: FullFloat> Quat<F>
{
    pub fn magnitude(&self) -> F {
        self.squared_magnitude().sqrt()
    }
}

impl<F: FullFloat> Quat<F> {
    pub fn is_normal(&self) -> bool {
        self.magnitude().approx_eq_ulps(
            &F::one(),
            <<F as ApproxEqUlps>::Flt as Ulps>::default_ulps()
        )
    }
}

// ----------------------------------------------------------------------------
// Add/Sub

impl<F: FullFloat> Add for Quat<F>
{
    type Output = Quat<F>;

    fn add(self, rhs: Quat<F>) -> Quat<F> {
        Quat {
            v: self.v + rhs.v,
            w: self.w + rhs.w,
        }
    }
}

impl<F: FullFloat> AddAssign for Quat<F>
{
    fn add_assign(&mut self, rhs: Quat<F>) {
        self.v += rhs.v;
        self.w += rhs.w;
    }
}

impl<F: FullFloat> Sub for Quat<F>
{
    type Output = Quat<F>;

    fn sub(self, rhs: Quat<F>) -> Quat<F> {
        Quat {
            v: self.v - rhs.v,
            w: self.w - rhs.w,
        }
    }
}

impl<F: FullFloat> SubAssign for Quat<F>
{
    fn sub_assign(&mut self, rhs: Quat<F>) {
        self.v -= rhs.v;
        self.w -= rhs.w;
    }
}

// ----------------------------------------------------------------------------
// Scalar Mul, Dot product, Hamiltonian product

impl<F: FullFloat> Mul<F> for Quat<F>
{
    type Output = Quat<F>;

    fn mul(self, rhs: F) -> Quat<F> {
        Quat {
            v: self.v * rhs,
            w: self.w * rhs,
        }
    }
}

impl<F: FullFloat> MulAssign<F> for Quat<F>
{
    fn mul_assign(&mut self, rhs: F) {
        self.v *= rhs;
        self.w *= rhs;
    }
}

impl<F: FullFloat> MulAssign for Quat<F>
{
    fn mul_assign(&mut self, rhs: Quat<F>) {
        self.v = self.v.cross(rhs.v)  +  rhs.v * self.w  +  self.v * rhs.w;
        self.w = self.w * rhs.w  -  self.v.dot(rhs.v);
    }
}

impl<F: FullFloat> Quat<F>
{
    pub fn dot(&self, other: Quat<F>) -> F
    {
        (self.w * other.w) + self.v.dot(other.v)
    }
}

impl<F: FullFloat> NQuat<F>
{
    pub fn dot(&self, other: Quat<F>) -> F
    {
        (self.w * other.w) + self.v.dot(other.v)
    }
}

impl<F: FullFloat> Mul for Quat<F>
{
    type Output = Quat<F>;

    fn mul(self, rhs: Quat<F>) -> Quat<F> {
        Quat {
            v: self.v.cross(rhs.v)  +  rhs.v * self.w  +  self.v * rhs.w,
            w: self.w * rhs.w  -  self.v.dot(rhs.v)
        }
    }
}

impl<F: FullFloat> Mul for NQuat<F>
{
    type Output = NQuat<F>;

    fn mul(self, rhs: NQuat<F>) -> NQuat<F> {
        let a: Quat<F> = From::from(self);
        let b: Quat<F> = From::from(rhs);
        From::from(a * b)
    }
}

// ----------------------------------------------------------------------------
// Conjugate

impl<F: FullFloat> Quat<F>
{
    pub fn conjugate(&self) -> Quat<F> {
        Quat {
            v: -self.v,
            w: self.w
        }
    }
}

impl<F: FullFloat> NQuat<F>
{
    pub fn conjugate(&self) -> NQuat<F> {
        NQuat::new_isnormal(
            -self.v,
            self.w)
    }
}

// ----------------------------------------------------------------------------
// Rotate a vector

impl<F: FullFloat> NQuat<F>
{
    // In general, the sandwich product qvq* does not care if Q is normalized or not.
    // However we presume it is normalized so we can take some shortcuts.
    // See Eric Lengyel, Foundations of Game Engine Development: Vol.1 Mathematics, p89
    pub fn rotate(&self, v: Vec3<F>) -> Vec3<F> {
        let dot = v.dot(self.v);

        v * ((self.w * self.w) - (self.v.x * self.v.x) - (self.v.y * self.v.y) - (self.v.z * self.v.z))
            + self.v * (dot + dot)
            + self.v.cross(v) * (self.w + self.w)
    }
}

// implemented also as NQuat * Vec3
impl<F: FullFloat> Mul<Vec3<F>> for NQuat<F>
{
    type Output = Vec3<F>;

    fn mul(self, rhs: Vec3<F>) -> Vec3<F> {
        self.rotate(rhs)
    }
}

// ----------------------------------------------------------------------------
// To/From Matrix

impl<F: FullFloat> From<NQuat<F>> for Mat3<F> {
    fn from(q: NQuat<F>) -> Mat3<F> {
        let two: F = NumCast::from(2.0_f32).unwrap();
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
            F::one() - two * (y2 + z2),             two * (xy - wz),            two * (xz + wy),
            two            * (xy + wz),  F::one() - two * (x2 + z2),            two * (yz - wx),
            two            * (xz - wy),             two * (yz + wx), F::one() - two * (x2 + y2)
        )
    }
}

impl<F: FullFloat> From<Mat3<F>> for NQuat<F> {
    fn from(m: Mat3<F>) -> NQuat<F> {
        let one: F = F::one();
        let half: F = NumCast::from(0.5_f32).unwrap();
        let quarter: F = NumCast::from(0.25_f32).unwrap();

        let sum = m.x.x + m.y.y + m.z.z;
        let x;
        let y;
        let z;
        let w;
        if sum>F::zero() {
            w = (sum + one).sqrt() * half;
            let f = quarter / w;
            x = (m.z.y - m.y.z) * f;
            y = (m.x.z - m.z.x) * f;
            z = (m.y.x - m.x.y) * f;
        }
        else if (m.x.x > m.y.y) && (m.x.x > m.z.z) {
            x = (m.x.x - m.y.y - m.z.z + one).sqrt() * half;
            let f = quarter / x;
            y = (m.y.x + m.x.y) * f;
            z = (m.x.z + m.z.x) * f;
            w = (m.z.y - m.y.z) * f;
        }
        else if m.y.y > m.z.z {
            y = (m.y.y - m.x.x - m.z.z + one).sqrt() * half;
            let f = quarter / y;
            x = (m.y.x + m.x.y) * f;
            z = (m.z.y + m.y.z) * f;
            w = (m.x.z - m.z.x) * f;
        }
        else {
            z = (m.z.z - m.x.x - m.y.y + one).sqrt() * half;
            let f = quarter / z;
            x = (m.x.z + m.z.x) * f;
            y = (m.z.y + m.y.z) * f;
            w = (m.y.x - m.x.y) * f;
        }

        NQuat::new_isnormal(Vec3 { x: x, y: y, z: z }, w)
    }
}

// ----------------------------------------------------------------------------
// ApproxEq

impl<F: FullFloat> ApproxEqUlps for Quat<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.v.approx_eq_ulps(&other.v, ulps) &&
            self.w.approx_eq_ulps(&other.w, ulps)
    }
}

impl<F: FullFloat> ApproxEqUlps for NQuat<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.v.approx_eq_ulps(&other.v, ulps) &&
            self.w.approx_eq_ulps(&other.w, ulps)
    }
}

// ----------------------------------------------------------------------------

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

    /*
    #[test]
    fn test_normal_or_not() {
        // This was for me to determine some things.
        let axis: Direction3<f32> = From::from(Vec3::<f32>::new(1.0, 5.0, 6.0003));
        let angle = Angle::<f32>::from_degrees(93.4);
        let q = NQuat::<f32>::from_axis_angle(&axis, &angle);

        if q.is_normal() {
            println!("from_axis_angle yields normal quats");
        } else {
            println!("from_axis_angle yields general quats");
        }
    }
    */
}
