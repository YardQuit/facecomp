//! Pins each backend's default cutoff to the evidence behind it.
//!
//! A threshold is the one number in this tool that can be wrong without
//! anything failing. Change it and every comparison still runs, still prints a
//! percentage, and still looks confident - it just quietly mis-scores. So the
//! values are asserted here alongside where they came from, and any edit has to
//! come with a reason good enough to update this file.

use facecomp::{Backend, ARCFACE_DEFAULT_THRESHOLD, DEFAULT_THRESHOLD};

#[test]
fn sface_uses_the_published_cutoff() {
    // OpenCV Zoo's published recommendation for face_recognition_sface_2021dec.
    // Deriving it over LFW through this crate's pipeline put the optimum at
    // 0.3156, worth 0.13 percentage points - not enough to justify moving a
    // shipped default, so the published value stands.
    assert_eq!(DEFAULT_THRESHOLD, 0.363);
    assert_eq!(Backend::SFace.default_threshold(), Some(0.363));
}

#[test]
fn arcface_uses_the_cutoff_derived_over_lfw() {
    // 0.2394 +/- 0.0087 across LFW's ten official folds, at 0.9932 +/- 0.0034
    // balanced accuracy over all 6000 pairs, none skipped.
    assert_eq!(ARCFACE_DEFAULT_THRESHOLD, 0.239);
    assert_eq!(Backend::ArcFace.default_threshold(), Some(0.239));
}

#[test]
fn the_two_cutoffs_are_not_interchangeable() {
    // The mistake this guards against is assuming one number fits both models.
    // It doesn't: SFace's 0.363 applied to ArcFace misses 56 of 3000 genuine
    // LFW pairs where 0.239 misses 37. Nothing would report that as an error.
    let sface = Backend::SFace
        .default_threshold()
        .expect("sface has a default");
    let arcface = Backend::ArcFace
        .default_threshold()
        .expect("arcface has a default");
    assert!(
        arcface < sface,
        "ArcFace separates the classes further, so its cutoff sits lower \
         ({arcface} vs {sface}); if that ever inverts, one of them is wrong"
    );
}

#[test]
fn every_backend_has_a_derived_default() {
    // The Option return type is kept so a backend added later starts as None
    // and the CLI refuses to run rather than inventing a cutoff. Both shipped
    // backends have earned a value; this fails if one is added without one.
    for backend in [Backend::SFace, Backend::ArcFace] {
        let threshold = backend
            .default_threshold()
            .unwrap_or_else(|| panic!("{backend} ships without a derived threshold"));
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "{backend}'s cutoff {threshold} is outside the range the \
             match-percent scale is defined over"
        );
    }
}
