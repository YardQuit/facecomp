//! Properties the multi-photo master template has to hold to.
//!
//! Averaging embeddings is only meaningful if every photo contributes equally
//! and the result stays on the unit hypersphere the cosine comparison assumes.
//! Both are easy to get subtly wrong in ways that don't error - they just tilt
//! the template toward whichever photo happened to embed with the largest
//! magnitude, which is not a property of the person.

use facecomp::{build_template, centroid, enrolment_agreement, FaceEncoding, Template};
use opencv::core::{Mat, MatTraitConst};

fn embedding(values: &[f32]) -> FaceEncoding {
    Mat::from_slice(values)
        .expect("1xN row")
        .try_clone()
        .expect("owned copy")
}

fn values(encoding: &FaceEncoding) -> Vec<f32> {
    (0..encoding.cols())
        .map(|i| *encoding.at_2d::<f32>(0, i).expect("element"))
        .collect()
}

fn norm(encoding: &FaceEncoding) -> f64 {
    values(encoding)
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn centroid_is_a_unit_vector() {
    // The comparison treats a dot product as a cosine, which is only true for
    // unit vectors. An un-normalised mean would silently rescale every score.
    let template = centroid(&[
        embedding(&[1.0, 0.0, 0.0]),
        embedding(&[0.0, 1.0, 0.0]),
        embedding(&[0.0, 0.0, 1.0]),
    ])
    .expect("centroid");
    assert!(
        (norm(&template) - 1.0).abs() < 1e-6,
        "centroid should be unit length, was {}",
        norm(&template)
    );
}

#[test]
fn every_photo_contributes_equally_regardless_of_magnitude() {
    // The property that makes this safe for either backend: SFace's raw
    // features are not normalised, so without normalising *before* averaging, a
    // photo that merely embedded with a larger magnitude would pull the
    // template toward itself. That isn't a fact about the person.
    let balanced = centroid(&[embedding(&[1.0, 0.0]), embedding(&[0.0, 1.0])]).expect("centroid");
    let lopsided = centroid(&[embedding(&[50.0, 0.0]), embedding(&[0.0, 1.0])]).expect("centroid");

    for (a, b) in values(&balanced).iter().zip(values(&lopsided).iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "scaling one input must not move the centroid: {balanced:?} vs {lopsided:?}",
            balanced = values(&balanced),
            lopsided = values(&lopsided)
        );
    }
}

#[test]
fn one_photo_passes_through_unchanged() {
    // Every run goes through this path, including the single-master case that
    // predates templates. It has to stay a no-op there, up to normalisation,
    // which the cosine comparison is invariant to.
    let template = centroid(&[embedding(&[3.0, 4.0])]).expect("centroid");
    let got = values(&template);
    assert!(
        (got[0] - 0.6).abs() < 1e-6 && (got[1] - 0.8).abs() < 1e-6,
        "got {got:?}"
    );
}

#[test]
fn agreement_reports_the_worst_pair() {
    // What guards against a stray photo of someone else in the enrolment set.
    let tight = enrolment_agreement(&[embedding(&[1.0, 0.0]), embedding(&[1.0, 0.0])])
        .expect("agreement")
        .expect("two photos");
    assert!(
        (tight - 1.0).abs() < 1e-6,
        "identical photos agree fully, got {tight}"
    );

    // Two matching photos plus an unrelated one: the odd photo out is what gets
    // reported, not the flattering average of all three pairs.
    let poisoned = enrolment_agreement(&[
        embedding(&[1.0, 0.0]),
        embedding(&[1.0, 0.0]),
        embedding(&[0.0, 1.0]),
    ])
    .expect("agreement")
    .expect("three photos");
    assert!(
        poisoned.abs() < 1e-6,
        "the disagreeing photo should set the score, got {poisoned}"
    );
}

#[test]
fn a_single_photo_has_nothing_to_cross_check() {
    let agreement = enrolment_agreement(&[embedding(&[1.0, 0.0])]).expect("agreement");
    assert_eq!(agreement, None);
}

#[test]
fn centroid_collapses_to_one_vector_and_max_keeps_them_all() {
    // The property the caller relies on to have a single code path: whichever
    // strategy is chosen, it gets back a list to score against and keeps the
    // best. Only the length differs.
    let photos = [
        embedding(&[1.0, 0.0]),
        embedding(&[0.0, 1.0]),
        embedding(&[0.0, -1.0]),
    ];
    assert_eq!(
        build_template(&photos, Template::Centroid)
            .expect("centroid")
            .len(),
        1
    );
    assert_eq!(
        build_template(&photos, Template::Max).expect("max").len(),
        3
    );
}

#[test]
fn max_preserves_each_photo_rather_than_blending_them() {
    // The whole reason max exists: two genuinely different looks stay
    // separately matchable instead of being averaged into one that resembles
    // neither. Scaled inputs to confirm they're normalised on the way out,
    // since the comparison treats a dot product as a cosine.
    let photos = [embedding(&[7.0, 0.0]), embedding(&[0.0, 2.0])];
    let template = build_template(&photos, Template::Max).expect("max");

    assert_eq!(values(&template[0]), vec![1.0, 0.0]);
    assert_eq!(values(&template[1]), vec![0.0, 1.0]);
    for vector in &template {
        assert!(
            (norm(vector) - 1.0).abs() < 1e-6,
            "each kept photo is unit length"
        );
    }

    // Averaging the same pair produces a vector matching neither original as
    // well as it matches itself - which is the trade-off, stated as a number.
    let blended = centroid(&photos).expect("centroid");
    assert!(
        (values(&blended)[0] - 0.70710678).abs() < 1e-6,
        "centroid sits between the two looks, at {:?}",
        values(&blended)
    );
}
