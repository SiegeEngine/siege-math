
use num_traits::identities::One;
use std::ops::{Deref, Sub, Add};
use super::{Vec2, Vec3, Vec4};

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


impl<F> Point2<F> {
    #[inline]
    pub fn new(x: F, y: F) -> Point2<F> {
        Point2(Vec2::new(x,y))
    }
}

impl<F> Point3<F> {
    #[inline]
    pub fn new(x: F, y: F, z: F) -> Point3<F> {
        Point3(Vec3::new(x,y,z))
    }
}

impl<F> Deref for Point2<F> {
    type Target = Vec2<F>;

    fn deref(&self) -> &Vec2<F>
    {
        &self.0
    }
}

impl<F> Deref for Point3<F> {
    type Target = Vec3<F>;

    fn deref(&self) -> &Vec3<F>
    {
        &self.0
    }
}


impl<F> From<Point2<F>> for Vec2<F> {
    fn from(v: Point2<F>) -> Vec2<F> {
        v.0
    }
}
impl<F> From<Point3<F>> for Vec3<F> {
    fn from(v: Point3<F>) -> Vec3<F> {
        v.0
    }
}
impl<F: One> From<Point3<F>> for Vec4<F> {
    fn from(v: Point3<F>) -> Vec4<F> {
        Vec4::new(v.0.x, v.0.y, v.0.z, F::one())
    }
}
impl<F> From<Vec2<F>> for Point2<F> {
    fn from(v: Vec2<F>) -> Point2<F> {
        Point2(v)
    }
}
impl<F> From<Vec3<F>> for Point3<F> {
    fn from(v: Vec3<F>) -> Point3<F> {
        Point3(v)
    }
}

impl Point3<f32> {
    #[allow(dead_code)]
    #[inline]
    fn from_vec4(v: Vec4<f32>) -> Option<Point3<f32>> {
        if v.w == 0.0 { return None; }
        Some(Point3(Vec3::new(v.x/v.w, v.y/v.w, v.z/v.w)))
    }
}
impl Point3<f64> {
    #[allow(dead_code)]
    #[inline]
    fn from_vec4(v: Vec4<f64>) -> Option<Point3<f64>> {
        if v.w == 0.0 { return None; }
        Some(Point3(Vec3::new(v.x/v.w, v.y/v.w, v.z/v.w)))
    }
}

impl<F: Add<Output=F>> Add<Vec2<F>> for Point2<F> {
    type Output = Point2<F>;

    #[inline]
    fn add(self, other: Vec2<F>) -> Point2<F> {
        Point2(self.0 + other)
    }
}
impl<F: Add<Output=F>> Add<Vec3<F>> for Point3<F> {
    type Output = Point3<F>;

    #[inline]
    fn add(self, other: Vec3<F>) -> Point3<F> {
        Point3(self.0 + other)
    }
}

// point - vector = point
impl<F: Sub<Output=F>> Sub<Vec2<F>> for Point2<F> {
    type Output = Point2<F>;

    #[inline]
    fn sub(self, other: Vec2<F>) -> Point2<F> {
        Point2(self.0 - other)
    }
}
impl<F: Sub<Output=F>> Sub<Vec3<F>> for Point3<F> {
    type Output = Point3<F>;

    #[inline]
    fn sub(self, other: Vec3<F>) -> Point3<F> {
        Point3(self.0 - other)
    }
}

// point - point = vector
impl<F: Sub<Output=F>> Sub<Point2<F>> for Point2<F> {
    type Output = Vec2<F>;

    #[inline]
    fn sub(self, other: Point2<F>) -> Vec2<F> {
        self.0 - other.0
    }
}
impl<F: Sub<Output=F>> Sub<Point3<F>> for Point3<F> {
    type Output = Vec3<F>;

    #[inline]
    fn sub(self, other: Point3<F>) -> Vec3<F> {
        self.0 - other.0
    }
}
