
use std::ops::{Deref, Sub, Add, Neg};
use float_cmp::{Ulps, ApproxEq};
use super::{Vec2, Vec3, Vec4};
use FullFloat;

/// Point vector in 2-dimensions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Point2<F>(pub Vec2<F>);

/// Point vector in 3-dimensions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Point3<F>(pub Vec3<F>);


impl<F: FullFloat> Point2<F> {
    #[inline]
    pub fn new(x: F, y: F) -> Point2<F> {
        Point2(Vec2::new(x,y))
    }
}

impl<F: FullFloat> Point3<F> {
    #[inline]
    pub fn new(x: F, y: F, z: F) -> Point3<F> {
        Point3(Vec3::new(x,y,z))
    }
}

// ----------------------------------------------------------------------------

impl<F: FullFloat> Deref for Point2<F> {
    type Target = Vec2<F>;

    fn deref(&self) -> &Vec2<F>
    {
        &self.0
    }
}

impl<F: FullFloat> Deref for Point3<F> {
    type Target = Vec3<F>;

    fn deref(&self) -> &Vec3<F>
    {
        &self.0
    }
}

// ----------------------------------------------------------------------------

impl<F: FullFloat> From<Point2<F>> for Vec2<F> {
    fn from(v: Point2<F>) -> Vec2<F> {
        v.0
    }
}
impl<F: FullFloat> From<Point3<F>> for Vec3<F> {
    fn from(v: Point3<F>) -> Vec3<F> {
        v.0
    }
}
impl<F: FullFloat> From<Point3<F>> for Vec4<F> {
    fn from(v: Point3<F>) -> Vec4<F> {
        Vec4::new(v.0.x, v.0.y, v.0.z, F::one())
    }
}
impl<F: FullFloat> From<Vec2<F>> for Point2<F> {
    fn from(v: Vec2<F>) -> Point2<F> {
        Point2(v)
    }
}
impl<F: FullFloat> From<Vec3<F>> for Point3<F> {
    fn from(v: Vec3<F>) -> Point3<F> {
        Point3(v)
    }
}
impl<F: FullFloat> From<Vec4<F>> for Point3<F> {
    fn from(v: Vec4<F>) -> Point3<F> {
        Point3(From::from(v))
    }
}

// ----------------------------------------------------------------------------

impl<F: FullFloat> Point3<F> {
    #[allow(dead_code)]
    #[inline]
    fn from_vec4(v: Vec4<F>) -> Option<Point3<F>> {
        if v.w == F::zero() { return None; }
        Some(Point3(Vec3::new(v.x/v.w, v.y/v.w, v.z/v.w)))
    }
}

// ----------------------------------------------------------------------------

impl<F: FullFloat> Add<Vec2<F>> for Point2<F> {
    type Output = Point2<F>;

    #[inline]
    fn add(self, other: Vec2<F>) -> Point2<F> {
        Point2(self.0 + other)
    }
}
impl<F: FullFloat> Add<Vec3<F>> for Point3<F> {
    type Output = Point3<F>;

    #[inline]
    fn add(self, other: Vec3<F>) -> Point3<F> {
        Point3(self.0 + other)
    }
}

// point - vector = point
impl<F: FullFloat> Sub<Vec2<F>> for Point2<F> {
    type Output = Point2<F>;

    #[inline]
    fn sub(self, other: Vec2<F>) -> Point2<F> {
        Point2(self.0 - other)
    }
}
impl<F: FullFloat> Sub<Vec3<F>> for Point3<F> {
    type Output = Point3<F>;

    #[inline]
    fn sub(self, other: Vec3<F>) -> Point3<F> {
        Point3(self.0 - other)
    }
}

// point - point = vector
impl<F: FullFloat> Sub<Point2<F>> for Point2<F> {
    type Output = Vec2<F>;

    #[inline]
    fn sub(self, other: Point2<F>) -> Vec2<F> {
        self.0 - other.0
    }
}
impl<F: FullFloat> Sub<Point3<F>> for Point3<F> {
    type Output = Vec3<F>;

    #[inline]
    fn sub(self, other: Point3<F>) -> Vec3<F> {
        self.0 - other.0
    }
}

// ----------------------------------------------------------------------------
// Neg

impl<F: FullFloat> Neg for Point3<F> {
    type Output = Point3<F>;

    #[inline]
    fn neg(self) -> Point3<F> {
        Point3(-self.0)
    }
}

// ----------------------------------------------------------------------------
// casting between float types

impl From<Point2<f64>> for Point2<f32> {
    fn from(p: Point2<f64>) -> Point2<f32> {
        Point2(From::from(p.0))
    }
}

impl From<Point2<f32>> for Point2<f64> {
    fn from(p: Point2<f32>) -> Point2<f64> {
        Point2(From::from(p.0))
    }
}

impl From<Point3<f64>> for Point3<f32> {
    fn from(p: Point3<f64>) -> Point3<f32> {
        Point3(From::from(p.0))
    }
}

impl From<Point3<f32>> for Point3<f64> {
    fn from(p: Point3<f32>) -> Point3<f64> {
        Point3(From::from(p.0))
    }
}

// ----------------------------------------------------------------------------
// ApproxEq

impl<F: FullFloat> ApproxEq for Point2<F> {
    type Flt = F;

    fn approx_eq(&self, other: &Self,
                 epsilon: <F as ApproxEq>::Flt,
                 ulps: <<F as ApproxEq>::Flt as Ulps>::U) -> bool
    {
        self.0.approx_eq(&other.0, epsilon, ulps)
    }
}

impl<F: FullFloat> ApproxEq for Point3<F> {
    type Flt = F;

    fn approx_eq(&self, other: &Self,
                 epsilon: <F as ApproxEq>::Flt,
                 ulps: <<F as ApproxEq>::Flt as Ulps>::U) -> bool
    {
        self.0.approx_eq(&other.0, epsilon, ulps)
    }
}
