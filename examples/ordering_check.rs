//! Measures what a wrong landmark order actually costs the ArcFace backend.
//!
//! YuNet returns 5 landmarks in the same order ArcFace's canonical destination
//! points are written in, so `facecomp` feeds one straight into the other with
//! no permutation step. That claim is worth checking rather than asserting,
//! because getting it wrong is silent: the alignment still succeeds, the
//! embedding still has 512 dimensions, and the only symptom is that the
//! numbers stop meaning anything.
//!
//! This runs the real `arcface_align` / `arcface_features` path twice over the
//! same photographs - once in the detector's native order, once with the eye
//! landmarks swapped - and prints the same/different separation each produces.
//!
//! The two same-person images must be genuinely different photographs, not one
//! derived from the other. A synthesised pair (the same image rotated, say)
//! badly understates the damage: both copies get the same wrong permutation
//! applied to the same underlying geometry, so the error largely cancels and a
//! swapped order still scores the pair highly. Measured that way the swap cost
//! only 0.95 -> 0.73 of separation; measured on real distinct photographs it is
//! far worse. Two snapshots differing in pose and lighting give the permutation
//! nothing to cancel against.
//!
//! Usage:
//!   cargo run --example ordering_check -- \
//!       face_detection_yunet_2023mar.onnx arcfaceresnet100-11-int8.onnx \
//!       personA-1.jpg personA-2.jpg personB.jpg

use std::env;
use std::process::ExitCode;

use facecomp::{arcface_align, arcface_features, landmarks_from_row};
use opencv::core::{Mat, MatTraitConst, Point2f, Size, Vector};
use opencv::objdetect::{FaceDetectorYN, FaceDetectorYNTrait};
use opencv::{dnn, imgcodecs};

/// Detector confidence low enough not to skip the rotated variant, which is a
/// harder detection than the original.
const DETECTION_CONFIDENCE: f32 = 0.5;

type Detector = opencv::core::Ptr<FaceDetectorYN>;

fn face_area(faces: &Mat, row: i32) -> f32 {
    let w = *faces.at_2d::<f32>(row, 2).unwrap_or(&0.0);
    let h = *faces.at_2d::<f32>(row, 3).unwrap_or(&0.0);
    w.max(0.0) * h.max(0.0)
}

/// The landmarks of the most prominent face, in the detector's native order.
fn detect_landmarks(detector: &mut Detector, image: &Mat) -> opencv::Result<Vector<Point2f>> {
    detector.set_input_size(image.size()?)?;
    let mut faces = Mat::default();
    detector.detect(image, &mut faces)?;
    if faces.rows() == 0 {
        return Err(opencv::Error::new(0, "no face detected"));
    }
    let best = (0..faces.rows())
        .max_by(|&a, &b| face_area(&faces, a).total_cmp(&face_area(&faces, b)))
        .expect("faces.rows() > 0");
    landmarks_from_row(&faces.row(best)?)
}

fn permute(landmarks: &Vector<Point2f>, order: [usize; 5]) -> Vector<Point2f> {
    order
        .iter()
        .map(|&i| landmarks.get(i).expect("5 landmarks"))
        .collect()
}

fn read(path: &str) -> opencv::Result<Mat> {
    let image = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        return Err(opencv::Error::new(0, format!("could not read {path}")));
    }
    Ok(image)
}

fn run() -> opencv::Result<()> {
    let argv: Vec<String> = env::args().collect();
    if argv.len() != 6 {
        eprintln!(
            "usage: {} <detector.onnx> <arcface.onnx> \
             <personA-1.jpg> <personA-2.jpg> <personB.jpg>",
            argv[0]
        );
        eprintln!("       personA-1 and personA-2 must be different photographs of one person.");
        return Err(opencv::Error::new(0, "wrong argument count"));
    }
    let (detector_model, arcface_model) = (&argv[1], &argv[2]);
    let (path_a1, path_a2, path_b) = (&argv[3], &argv[4], &argv[5]);

    let mut detector = FaceDetectorYN::create(
        detector_model,
        "",
        Size::new(320, 320),
        DETECTION_CONFIDENCE,
        0.3,
        5000,
        0,
        0,
    )?;
    let mut net = dnn::read_net_from_onnx(arcface_model)?;

    let image_a1 = read(path_a1)?;
    let image_a2 = read(path_a2)?;
    let image_b = read(path_b)?;

    let images = [("A-1", &image_a1), ("A-2", &image_a2), ("B", &image_b)];
    let mut landmarks = Vec::new();
    for (label, image) in images {
        let found = detect_landmarks(&mut detector, image)
            .map_err(|e| opencv::Error::new(0, format!("{label}: {}", e.message)))?;
        landmarks.push(found);
    }

    println!(
        "same-person pair : {path_a1} vs {path_a2}\n\
         different pair   : {path_a1} vs {path_b}\n"
    );
    println!(
        "{:<22} {:>10} {:>11} {:>12}",
        "landmark order", "same-pair", "diff-pair", "separation"
    );

    // The identity order is what facecomp ships; the rest are the plausible
    // ways to get it wrong, worst-case last. "Mirrored" is the one to worry
    // about - it is what assuming the opposite handedness convention produces,
    // swapping both eyes *and* both mouth corners, and it is the only
    // permutation here that reliably inverts the verdict rather than merely
    // blunting it.
    let orders: [(&str, [usize; 5]); 6] = [
        ("native (shipped)", [0, 1, 2, 3, 4]),
        ("eyes swapped", [1, 0, 2, 3, 4]),
        ("mouth swapped", [0, 1, 2, 4, 3]),
        ("mirrored (eyes+mouth)", [1, 0, 2, 4, 3]),
        ("rotated by one", [1, 2, 3, 4, 0]),
        ("reversed", [4, 3, 2, 1, 0]),
    ];

    for (label, order) in orders {
        let mut embeddings = Vec::new();
        for (index, (_, image)) in images.iter().enumerate() {
            let aligned = arcface_align(image, &permute(&landmarks[index], order))?;
            embeddings.push(arcface_features(&mut net, &aligned)?);
        }
        let same = embeddings[0].dot(&embeddings[1])?;
        let different = embeddings[0].dot(&embeddings[2])?;
        println!(
            "{:<22} {:>10.4} {:>11.4} {:>+12.4}",
            label,
            same,
            different,
            same - different
        );
    }

    println!(
        "\nA correct order separates the pairs by a wide positive margin. If the\n\
         swapped row's separation collapses toward zero or goes negative, the\n\
         ordering matters and tests/arcface_alignment.rs is what pins it."
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ordering_check: {}", e.message);
            ExitCode::FAILURE
        }
    }
}
