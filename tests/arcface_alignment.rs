//! Pins the landmark ordering the ArcFace backend depends on.
//!
//! YuNet emits its 5 landmarks in the same order `ARCFACE_DST` is written in,
//! so `facecomp` maps one onto the other directly. Nothing about a wrong order
//! fails loudly: alignment still succeeds, the embedding still comes back with
//! 512 dimensions, and the only symptom is that the similarities stop meaning
//! anything. Measured on real photographs with the shipped int8 model, feeding
//! the landmarks in mirrored order (both eyes swapped and both mouth corners
//! swapped, which is what assuming the opposite handedness convention gives
//! you) scored a *different* person at 0.5731 against the same person's 0.4603
//! - an inverted verdict, reported with no hint that anything was wrong.
//!
//! These tests need no model and no photographs. They work on the alignment
//! transform itself: fit it to landmarks that are a known similarity transform
//! of the canonical destination points, and the residual is ~0 only when the
//! two are paired up in the right order. `examples/ordering_check.rs` is the
//! companion that measures the same thing end-to-end on real images.

use facecomp::{arcface_dst_points, arcface_transform, landmarks_from_row, ARCFACE_DST};
use opencv::core::{Mat, MatTraitConst, Point2f, Vector};

/// Largest per-landmark error, in pixels, that still counts as an exact fit.
///
/// The fit is over exact synthetic inputs, so the only error is floating point;
/// the real residuals come out around 1e-5.
const EXACT: f64 = 1e-3;

/// Applies a 2x3 affine matrix (as `estimateAffinePartial2D` returns) to a point.
fn apply(transform: &Mat, p: Point2f) -> Point2f {
    let at = |r: i32, c: i32| *transform.at_2d::<f64>(r, c).expect("2x3 matrix");
    let (x, y) = (f64::from(p.x), f64::from(p.y));
    Point2f::new(
        (at(0, 0) * x + at(0, 1) * y + at(0, 2)) as f32,
        (at(1, 0) * x + at(1, 1) * y + at(1, 2)) as f32,
    )
}

/// Builds landmarks by putting `ARCFACE_DST` through a known rotation, uniform
/// scale and translation - the exact class of transform the aligner inverts.
fn synthetic_landmarks(angle_deg: f64, scale: f64, tx: f64, ty: f64) -> Vector<Point2f> {
    let radians = angle_deg.to_radians();
    let (cos, sin) = (scale * radians.cos(), scale * radians.sin());
    ARCFACE_DST
        .iter()
        .map(|[x, y]| {
            let (x, y) = (f64::from(*x), f64::from(*y));
            Point2f::new(
                (cos * x - sin * y + tx) as f32,
                (sin * x + cos * y + ty) as f32,
            )
        })
        .collect()
}

fn permute(landmarks: &Vector<Point2f>, order: [usize; 5]) -> Vector<Point2f> {
    order
        .iter()
        .map(|&i| landmarks.get(i).expect("5 landmarks"))
        .collect()
}

/// Worst per-landmark distance between `transform(landmarks)` and where
/// `ARCFACE_DST` says each one should have landed.
fn worst_residual(landmarks: &Vector<Point2f>) -> f64 {
    let transform = arcface_transform(landmarks).expect("transform is estimable");
    let destination = arcface_dst_points();
    (0..landmarks.len())
        .map(|i| {
            let got = apply(&transform, landmarks.get(i).expect("landmark"));
            let want = destination.get(i).expect("destination");
            f64::from((got.x - want.x).hypot(got.y - want.y))
        })
        .fold(0.0, f64::max)
}

#[test]
fn native_order_recovers_the_alignment_exactly() {
    // A few different poses, so this can't pass by luck on one benign case.
    for (angle, scale, tx, ty) in [
        (0.0, 1.0, 0.0, 0.0),
        (15.0, 1.7, 40.0, -25.0),
        (-30.0, 0.6, -12.0, 80.0),
        (170.0, 2.4, 300.0, 150.0),
    ] {
        let landmarks = synthetic_landmarks(angle, scale, tx, ty);
        let residual = worst_residual(&landmarks);
        assert!(
            residual < EXACT,
            "native order should invert a {angle} deg / {scale}x / ({tx},{ty}) pose exactly, \
             but the worst landmark was off by {residual:.6}px"
        );
    }
}

#[test]
fn wrong_orders_destroy_the_alignment() {
    // Every permutation that has a plausible story behind it. "mouth swapped"
    // is included precisely because it is the mildest: it is nearly harmless
    // end-to-end, which is what makes a residual check the honest guard rather
    // than eyeballing a similarity score.
    let wrong: [(&str, [usize; 5]); 5] = [
        ("eyes swapped", [1, 0, 2, 3, 4]),
        ("mouth swapped", [0, 1, 2, 4, 3]),
        ("mirrored (eyes+mouth)", [1, 0, 2, 4, 3]),
        ("rotated by one", [1, 2, 3, 4, 0]),
        ("reversed", [4, 3, 2, 1, 0]),
    ];

    let landmarks = synthetic_landmarks(15.0, 1.7, 40.0, -25.0);
    for (label, order) in wrong {
        let residual = worst_residual(&permute(&landmarks, order));
        assert!(
            residual > 1.0,
            "{label} should not fit the canonical layout, yet its worst landmark \
             was only {residual:.6}px out - the ordering contract is not being enforced"
        );
    }
}

#[test]
fn destination_points_are_the_canonical_ones() {
    // Guards the constant itself: the tests above would still pass if every row
    // moved together, since a similarity transform would absorb it.
    assert_eq!(
        ARCFACE_DST,
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        "ArcFace's canonical 112x112 destination landmarks changed"
    );

    // Eyes above nose above mouth, and each pair left-to-right. This is the
    // structure the ordering contract actually rests on, spelled out so a
    // reordering is a test failure with an obvious cause rather than a number
    // that no longer matches.
    let [right_eye, left_eye, nose, right_mouth, left_mouth] = ARCFACE_DST;
    assert!(right_eye[0] < left_eye[0], "row 0 is the right eye, row 1 the left");
    assert!(right_mouth[0] < left_mouth[0], "row 3 is the right mouth corner, row 4 the left");
    assert!(right_eye[1] < nose[1], "eyes sit above the nose");
    assert!(nose[1] < right_mouth[1], "the nose sits above the mouth");
}

#[test]
fn landmarks_are_read_straight_out_of_the_detection_row() {
    // A YuNet row is [x, y, w, h, 5x(lm x, lm y), score]. Numbering the columns
    // makes the mapping self-evident: landmark i must come back as
    // (4 + 2i, 5 + 2i), with no permutation applied on the way out.
    let columns: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let row = Mat::from_slice(&columns).expect("1x15 row");

    let landmarks = landmarks_from_row(&row).expect("landmarks are readable");

    assert_eq!(landmarks.len(), 5);
    for i in 0..5 {
        let got = landmarks.get(i).expect("landmark");
        let expected = Point2f::new((4 + 2 * i) as f32, (5 + 2 * i) as f32);
        assert_eq!(
            (got.x, got.y),
            (expected.x, expected.y),
            "landmark {i} should be read from columns {} and {}",
            4 + 2 * i,
            5 + 2 * i
        );
    }
}
