
use {Point3, NQuat};

pub struct Position<F> {
    pub point: Point3<F>,
    pub ori: NQuat<F>,
}
