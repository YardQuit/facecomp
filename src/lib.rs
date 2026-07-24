//! Core face-comparison logic: detect a face, embed it, and compare embeddings.
//!
//! This crate is the shared backend for the `facecomp` CLI and for the Emacs
//! frontend in `emacs/facecomp.el`, which just shells out to that CLI.
//!
//! Detection uses OpenCV's YuNet (`FaceDetectorYN`); embedding and comparison
//! use OpenCV's SFace (`FaceRecognizerSF`). SFace's alignment step needs the
//! 5-point landmarks YuNet already returns alongside each bounding box, so
//! both stages come from OpenCV rather than mixing detector implementations.

use std::fmt;
use std::path::{Path, PathBuf};

use opencv::core::{Mat, MatTraitConst, Size};
use opencv::objdetect::{
    FaceDetectorYN, FaceDetectorYNTrait, FaceRecognizerSF, FaceRecognizerSFTrait,
    FaceRecognizerSFTraitConst, FaceRecognizerSF_FR_COSINE,
};
use opencv::{imgcodecs, Error as CvError};

/// Cosine similarity at/above which SFace's model considers two faces the
/// same person.
///
/// This is the threshold OpenCV Zoo publishes for the
/// `face_recognition_sface_2021dec` model, not something we derived
/// ourselves.
pub const DEFAULT_THRESHOLD: f64 = 0.363;

/// A face embedding: a 1x128 row produced by `FaceRecognizerSF::feature`.
pub type FaceEncoding = Mat;

#[derive(Debug)]
pub enum FacecompError {
    Image(PathBuf, String),
    Model(String),
    NoFaceDetected(PathBuf),
}

impl fmt::Display for FacecompError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FacecompError::Image(path, e) => {
                write!(f, "failed to read image {}: {e}", path.display())
            }
            FacecompError::Model(e) => write!(f, "failed to load model: {e}"),
            FacecompError::NoFaceDetected(path) => {
                write!(f, "no face detected in {}", path.display())
            }
        }
    }
}

impl std::error::Error for FacecompError {}

/// Loads OpenCV's YuNet detector and SFace recognizer once and reuses them
/// across many comparisons.
pub struct FaceComparer {
    detector: opencv::core::Ptr<FaceDetectorYN>,
    recognizer: opencv::core::Ptr<FaceRecognizerSF>,
}

impl FaceComparer {
    /// `detector_model` and `recognizer_model` are paths to OpenCV Zoo's
    /// `face_detection_yunet_2023mar.onnx` and
    /// `face_recognition_sface_2021dec.onnx` respectively.
    pub fn new(
        detector_model: impl AsRef<Path>,
        recognizer_model: impl AsRef<Path>,
    ) -> Result<Self, FacecompError> {
        let detector_model = path_to_str(detector_model.as_ref())?;
        let recognizer_model = path_to_str(recognizer_model.as_ref())?;

        // The real input size is set per-image in `detect`; this initial
        // size is just a placeholder required by the constructor.
        let detector = FaceDetectorYN::create(
            detector_model,
            "",
            Size::new(320, 320),
            0.9,
            0.3,
            5000,
            0,
            0,
        )
        .map_err(|e| FacecompError::Model(e.to_string()))?;
        let recognizer = FaceRecognizerSF::create(recognizer_model, "", 0, 0)
            .map_err(|e| FacecompError::Model(e.to_string()))?;

        Ok(Self {
            detector,
            recognizer,
        })
    }

    /// Detects the most prominent face in the image at `path` (the one with
    /// the largest bounding box) and returns its 128-dimension embedding.
    pub fn encode_face(&mut self, path: impl AsRef<Path>) -> Result<FaceEncoding, FacecompError> {
        let path = path.as_ref();
        let (image, faces) = self.detect(path)?;
        let index = largest_face_index(&faces)
            .ok_or_else(|| FacecompError::NoFaceDetected(path.to_path_buf()))?;
        self.encode_row(&image, &faces, index, path)
    }

    /// Detects every face in the image at `path` and returns one embedding
    /// per face, in the order YuNet found them. Useful for photos with more
    /// than one person in frame, where the caller wants to compare against
    /// each face rather than assume there's only one.
    pub fn encode_all_faces(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<FaceEncoding>, FacecompError> {
        let path = path.as_ref();
        let (image, faces) = self.detect(path)?;
        (0..faces.rows())
            .map(|index| self.encode_row(&image, &faces, index, path))
            .collect()
    }

    /// Compares two embeddings by cosine similarity, as SFace expects.
    pub fn compare(
        &self,
        a: &FaceEncoding,
        b: &FaceEncoding,
        threshold: f64,
    ) -> Result<Comparison, FacecompError> {
        let similarity = self
            .recognizer
            .match_(a, b, FaceRecognizerSF_FR_COSINE)
            .map_err(|e| FacecompError::Model(e.to_string()))?;
        Ok(Comparison {
            similarity,
            match_percent: similarity_to_percent(similarity, threshold),
        })
    }

    /// Runs YuNet detection on the image at `path`. `faces` is a Mat with one
    /// row per detected face: `[x, y, w, h, 5x(landmark x, landmark y), score]`.
    fn detect(&mut self, path: &Path) -> Result<(Mat, Mat), FacecompError> {
        let path_str = path_to_str(path)?;
        let image = imgcodecs::imread(path_str, imgcodecs::IMREAD_COLOR)
            .map_err(|e| FacecompError::Image(path.to_path_buf(), e.to_string()))?;
        if image.empty() {
            return Err(FacecompError::Image(
                path.to_path_buf(),
                "file not found or not a decodable image".to_string(),
            ));
        }

        let size = image.size().map_err(|e| cv_image_err(path, e))?;
        self.detector
            .set_input_size(size)
            .map_err(|e| cv_image_err(path, e))?;

        let mut faces = Mat::default();
        self.detector
            .detect(&image, &mut faces)
            .map_err(|e| cv_image_err(path, e))?;
        if faces.rows() == 0 {
            return Err(FacecompError::NoFaceDetected(path.to_path_buf()));
        }

        Ok((image, faces))
    }

    /// Aligns and embeds the face in row `index` of `faces` (as detected by
    /// `detect`) from the source `image`.
    fn encode_row(
        &mut self,
        image: &Mat,
        faces: &Mat,
        index: i32,
        path: &Path,
    ) -> Result<FaceEncoding, FacecompError> {
        let face_row = faces.row(index).map_err(|e| cv_image_err(path, e))?;

        let mut aligned = Mat::default();
        self.recognizer
            .align_crop(image, &face_row, &mut aligned)
            .map_err(|e| cv_image_err(path, e))?;

        let mut feature = Mat::default();
        self.recognizer
            .feature(&aligned, &mut feature)
            .map_err(|e| cv_image_err(path, e))?;
        feature.try_clone().map_err(|e| cv_image_err(path, e))
    }
}

fn path_to_str(path: &Path) -> Result<&str, FacecompError> {
    path.to_str().ok_or_else(|| {
        FacecompError::Model(format!("{}: path is not valid UTF-8", path.display()))
    })
}

fn cv_image_err(path: &Path, e: CvError) -> FacecompError {
    FacecompError::Image(path.to_path_buf(), e.to_string())
}

fn largest_face_index(faces: &Mat) -> Option<i32> {
    (0..faces.rows()).max_by(|&a, &b| face_area(faces, a).total_cmp(&face_area(faces, b)))
}

fn face_area(faces: &Mat, row: i32) -> f32 {
    let w = *faces.at_2d::<f32>(row, 2).unwrap_or(&0.0);
    let h = *faces.at_2d::<f32>(row, 3).unwrap_or(&0.0);
    w.max(0.0) * h.max(0.0)
}

#[derive(Debug, Clone, Copy)]
pub struct Comparison {
    pub similarity: f64,
    pub match_percent: f64,
}

/// Maps an SFace cosine similarity onto a 0-100 heuristic "percent match".
///
/// This is not a calibrated probability - OpenCV Zoo only publishes the
/// 0.363 same/different-person cutoff, not a similarity-to-confidence curve.
/// We linearly scale similarity so that 1.0 (identical) -> 100% and the
/// threshold's mirror image below it -> 0%, which puts the same/different
/// cutoff itself at exactly 50%.
pub fn similarity_to_percent(similarity: f64, threshold: f64) -> f64 {
    let floor = 2.0 * threshold - 1.0;
    let percent = 100.0 * (similarity - floor) / (1.0 - floor);
    percent.clamp(0.0, 100.0)
}

/// Maps a match percentage onto the standard intelligence-community words-of-
/// estimative-probability yardstick (the same bands used in ICD 203), so a
/// reader gets a qualitative call alongside the raw number.
///
/// Publisher: Office of the Director of National Intelligence (ODNI).
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
