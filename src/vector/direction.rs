
use num_traits::identities::Zero;
use num_traits::Float;
use std::ops::{Deref, Mul, Sub, Add, Neg};
use float_cmp::{Ulps, ApproxEqUlps};
use super::{Vec2, Vec3, Vec4};
use Angle;

/// Direction vector in 2-dimensions (normalized)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Direction2<F>(Vec2<F>);

/// Direction vector in 3-dimensions (normalized)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Direction3<F>(Vec3<F>);

pub const X_AXIS_F32: Direction3<f32> = Direction3::<f32>(
    Vec3::<f32> { x: 1.0, y: 0.0, z: 0.0 }
);
pub const Y_AXIS_F32: Direction3<f32> = Direction3::<f32>(
    Vec3::<f32> { x: 0.0, y: 1.0, z: 0.0 }
);
pub const Z_AXIS_F32: Direction3<f32> = Direction3::<f32>(
    Vec3::<f32> { x: 0.0, y: 0.0, z: 1.0 }
);
pub const X_AXIS_F64: Direction3<f64> = Direction3::<f64>(
    Vec3::<f64> { x: 1.0, y: 0.0, z: 0.0 }
);
pub const Y_AXIS_F64: Direction3<f64> = Direction3::<f64>(
    Vec3::<f64> { x: 0.0, y: 1.0, z: 0.0 }
);
pub const Z_AXIS_F64: Direction3<f64> = Direction3::<f64>(
    Vec3::<f64> { x: 0.0, y: 0.0, z: 1.0 }
);

impl<F> Direction2<F> {
    #[inline]
    pub fn new_isnormal(x: F, y: F) -> Direction2<F> {
        Direction2(Vec2::new(x,y))
    }
}

impl<F> Direction3<F> {
    #[inline]
    pub fn new_isnormal(x: F, y: F, z: F) -> Direction3<F> {
        Direction3(Vec3::new(x,y,z))
    }
}

impl<F> Deref for Direction2<F> {
    type Target = Vec2<F>;

    fn deref(&self) -> &Vec2<F>
    {
        &self.0
    }
}

impl<F> Deref for Direction3<F> {
    type Target = Vec3<F>;

    fn deref(&self) -> &Vec3<F>
    {
        &self.0
    }
}

impl<F> From<Direction2<F>> for Vec2<F> {
    fn from(v: Direction2<F>) -> Vec2<F> {
        v.0
    }
}
impl<F> From<Direction3<F>> for Vec3<F> {
    fn from(v: Direction3<F>) -> Vec3<F> {
        v.0
    }
}
impl<F: Zero> From<Direction3<F>> for Vec4<F> {
    fn from(v: Direction3<F>) -> Vec4<F> {
        Vec4::new(v.0.x, v.0.y, v.0.z, F::zero())
    }
}

impl From<Vec2<f32>> for Direction2<f32> {
    fn from(mut v: Vec2<f32>) -> Direction2<f32> {
        let mag = v.magnitude();
        v.x /= mag;
        v.y /= mag;
        Direction2(v)
    }
}
impl From<Vec2<f64>> for Direction2<f64> {
    fn from(mut v: Vec2<f64>) -> Direction2<f64> {
        let mag = v.magnitude();
        v.x /= mag;
        v.y /= mag;
        Direction2(v)
    }
}
impl From<Vec3<f32>> for Direction3<f32> {
    fn from(mut v: Vec3<f32>) -> Direction3<f32> {
        let mag = v.magnitude();
        v.x /= mag;
        v.y /= mag;
        v.z /= mag;
        Direction3(v)
    }
}
impl From<Vec3<f64>> for Direction3<f64> {
    fn from(mut v: Vec3<f64>) -> Direction3<f64> {
        let mag = v.magnitude();
        v.x /= mag;
        v.y /= mag;
        v.z /= mag;
        Direction3(v)
    }
}

impl Direction3<f32> {
    #[allow(dead_code)]
    #[inline]
    fn from_vec4(v: Vec4<f32>) -> Option<Direction3<f32>> {
        if v.w != 0.0 { return None; }
        Some(Direction3(v.truncate_w()))
    }
}
impl Direction3<f64> {
    #[allow(dead_code)]
    #[inline]
    fn from_vec4(v: Vec4<f64>) -> Option<Direction3<f64>> {
        if v.w != 0.0 { return None; }
        Some(Direction3(v.truncate_w()))
    }
}

impl<F: Copy + Mul<F,Output=F> + Sub<F,Output=F>> Direction3<F> {
    #[inline]
    pub fn cross(&self, rhs: Direction3<F>) -> Direction3<F> {
        Direction3(self.0.cross(rhs.0))
    }
}

impl<F: Copy + Mul<F,Output=F> + Add<F,Output=F>> Direction3<F> {
    #[inline]
    pub fn dot(&self, rhs: Direction3<F>) -> F {
        self.0.dot(rhs.0)
    }
}

impl<F: Neg<Output=F>> Neg for Direction3<F> {
    type Output = Direction3<F>;

    #[inline]
    fn neg(self) -> Direction3<F> {
        Direction3(-self.0)
    }
}

impl<F: Ulps + ApproxEqUlps<Flt=F>> ApproxEqUlps for Direction2<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.0.approx_eq_ulps(&other.0, ulps)
    }
}

impl<F: Ulps + ApproxEqUlps<Flt=F>> ApproxEqUlps for Direction3<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.0.approx_eq_ulps(&other.0, ulps)
    }
}

impl<F: Copy + Float> Direction3<F> {
    pub fn from_lat_long(latitude: Angle<F>, longitude: Angle<F>) -> Direction3<F>
    {
        let (slat,clat) = latitude.as_radians().sin_cos();
        let (slon,clon) = longitude.as_radians().sin_cos();
        Direction3(Vec3 {
            x: clat * slon,
            y: slat,
            z: slat * clon,
        })
    }

    pub fn to_lat_long(&self) -> (Angle<F>, Angle<F>) {
        let lat = self.0.y.acos();
        let lon = (self.0.z / self.0.y).acos();
        (Angle::from_radians(lat), Angle::from_radians(lon))
    }
}
