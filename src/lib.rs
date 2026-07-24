//! Core face-comparison logic: detect a face, embed it, and compare embeddings.
//!
//! This crate is the shared backend for the `facecomp` CLI and for the Emacs
//! frontend in `emacs/facecomp.el`, which just shells out to that CLI.

use std::fmt;
use std::path::{Path, PathBuf};

use dlib_face_recognition::{
    FaceDetector, FaceDetectorTrait, FaceEncoderNetwork, FaceEncoderTrait, FaceEncoding,
    FaceEncodings, FaceLocations, ImageMatrix, LandmarkPredictor, LandmarkPredictorTrait,
    Rectangle,
};

/// Euclidean distance at/below which dlib's model considers two faces the same person.
///
/// This is dlib's own published recommendation for its ResNet face-encoding
/// network, not something we derived ourselves.
pub const DEFAULT_THRESHOLD: f64 = 0.6;

#[derive(Debug)]
pub enum FacecompError {
    Image(image::ImageError),
    Model(String),
    NoFaceDetected(PathBuf),
}

impl fmt::Display for FacecompError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FacecompError::Image(e) => write!(f, "failed to read image: {e}"),
            FacecompError::Model(e) => write!(f, "failed to load model: {e}"),
            FacecompError::NoFaceDetected(path) => {
                write!(f, "no face detected in {}", path.display())
            }
        }
    }
}

impl std::error::Error for FacecompError {}

impl From<image::ImageError> for FacecompError {
    fn from(e: image::ImageError) -> Self {
        FacecompError::Image(e)
    }
}

/// Loads dlib's models once and reuses them across many comparisons.
pub struct FaceComparer {
    detector: FaceDetector,
    predictor: LandmarkPredictor,
    encoder: FaceEncoderNetwork,
}

impl FaceComparer {
    /// `landmark_model` and `encoder_model` are paths to dlib's
    /// `shape_predictor_68_face_landmarks.dat` and
    /// `dlib_face_recognition_resnet_model_v1.dat` respectively.
    pub fn new(
        landmark_model: impl AsRef<Path>,
        encoder_model: impl AsRef<Path>,
    ) -> Result<Self, FacecompError> {
        let predictor =
            LandmarkPredictor::open(landmark_model.as_ref()).map_err(FacecompError::Model)?;
        let encoder =
            FaceEncoderNetwork::open(encoder_model.as_ref()).map_err(FacecompError::Model)?;
        Ok(Self {
            detector: FaceDetector::new(),
            predictor,
            encoder,
        })
    }

    /// Detects the most prominent face in the image at `path` (the one with
    /// the largest bounding box) and returns its 128-dimension embedding.
    pub fn encode_face(&self, path: impl AsRef<Path>) -> Result<FaceEncoding, FacecompError> {
        let path = path.as_ref();
        let (faces, encodings) = self.detect(path)?;
        let index = largest_face_index(&faces)
            .ok_or_else(|| FacecompError::NoFaceDetected(path.to_path_buf()))?;
        Ok(encodings[index].clone())
    }

    /// Detects every face in the image at `path` and returns one embedding
    /// per face, in the order dlib's detector found them. Useful for photos
    /// with more than one person in frame, where the caller wants to compare
    /// against each face rather than assume there's only one.
    pub fn encode_all_faces(&self, path: impl AsRef<Path>) -> Result<Vec<FaceEncoding>, FacecompError> {
        let (_, encodings) = self.detect(path.as_ref())?;
        Ok(encodings.to_vec())
    }

    fn detect(&self, path: &Path) -> Result<(FaceLocations, FaceEncodings), FacecompError> {
        let image = image::open(path)?.to_rgb8();
        let matrix = ImageMatrix::from_image(&image);

        let faces = self.detector.face_locations(&matrix);
        if faces.is_empty() {
            return Err(FacecompError::NoFaceDetected(path.to_path_buf()));
        }

        let landmarks: Vec<_> = faces
            .iter()
            .map(|face| self.predictor.face_landmarks(&matrix, face))
            .collect();
        let encodings = self.encoder.get_face_encodings(&matrix, &landmarks, 0);
        Ok((faces, encodings))
    }
}

fn largest_face_index(faces: &[Rectangle]) -> Option<usize> {
    faces
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| {
            let width = (r.right - r.left).max(0) as i64;
            let height = (r.bottom - r.top).max(0) as i64;
            width * height
        })
        .map(|(index, _)| index)
}

#[derive(Debug, Clone, Copy)]
pub struct Comparison {
    pub distance: f64,
    pub match_percent: f64,
    pub same_person: bool,
}

pub fn compare(a: &FaceEncoding, b: &FaceEncoding, threshold: f64) -> Comparison {
    let distance = a.distance(b);
    Comparison {
        distance,
        match_percent: distance_to_percent(distance, threshold),
        same_person: distance <= threshold,
    }
}

/// Maps a dlib embedding distance onto a 0-100 heuristic "percent match".
///
/// This is not a calibrated probability - dlib only publishes the 0.6
/// same/different-person cutoff, not a distance-to-confidence curve. We
/// linearly scale distance so that 0 -> 100% and 2x the threshold -> 0%,
/// which puts the same/different cutoff itself at exactly 50%.
pub fn distance_to_percent(distance: f64, threshold: f64) -> f64 {
    let percent = 100.0 * (1.0 - distance / (2.0 * threshold));
    percent.clamp(0.0, 100.0)
}

/// Maps a match percentage onto the standard intelligence-community words-of-
/// estimative-probability yardstick (the same bands used in ICD 203), so a
/// reader gets a qualitative call alongside the raw number.
pub fn confidence_label(match_percent: f64) -> &'static str {
    match match_percent {
        p if p >= 95.0 => "Almost certain",
        p if p >= 80.0 => "Very likely",
        p if p >= 55.0 => "Likely",
        p if p >= 45.0 => "Even chance",
        p if p >= 20.0 => "Unlikely",
        p if p >= 5.0 => "Very unlikely",
        _ => "Almost no chance",
    }
}
