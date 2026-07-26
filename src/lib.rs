//! Core face-comparison logic: detect a face, embed it, and compare embeddings.
//!
//! This crate is the shared backend for the `facecomp` CLI and for the Emacs
//! frontend in `emacs/facecomp.el`, which just shells out to that CLI.
//!
//! Detection uses OpenCV's YuNet (`FaceDetectorYN`). Embedding is pluggable -
//! see [`Backend`] - but either way it consumes the 5-point landmarks YuNet
//! already returns alongside each bounding box, so detection and alignment
//! agree rather than mixing detector implementations.

use std::fmt;
use std::path::{Path, PathBuf};

use opencv::core::{
    Mat, MatTraitConst, Point2f, Scalar, Size, Vector, CV_32F, NORM_L2,
};
use opencv::objdetect::{
    FaceDetectorYN, FaceDetectorYNTrait, FaceRecognizerSF, FaceRecognizerSFTrait,
    FaceRecognizerSFTraitConst, FaceRecognizerSF_FR_COSINE,
};
use opencv::{calib3d, core as cv_core, dnn, imgcodecs, imgproc, Error as CvError};

use dnn::NetTrait;

/// Cosine similarity at/above which SFace's model considers two faces the
/// same person.
///
/// This is the threshold OpenCV Zoo publishes for the
/// `face_recognition_sface_2021dec` model, not something we derived
/// ourselves.
pub const DEFAULT_THRESHOLD: f64 = 0.363;

/// Which recognition model turns an aligned face into an embedding.
///
/// The two differ in more than the weights: SFace bundles its own alignment
/// (`FaceRecognizerSF::alignCrop`) and similarity (`FaceRecognizerSF::match`),
/// whereas ArcFace is a bare ONNX graph that has to be fed a face this crate
/// aligns itself. See [`FaceComparer::encode_row`] for that path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// OpenCV Zoo's `face_recognition_sface_2021dec.onnx`; 128-d embeddings.
    #[default]
    SFace,
    /// ONNX Model Zoo's `arcfaceresnet100-11-int8.onnx`; 512-d embeddings.
    ArcFace,
}

impl Backend {
    /// The same/different-person cutoff to use when the caller didn't pick one.
    ///
    /// `None` means we do not have a trustworthy value and the caller must
    /// supply `--threshold` explicitly. That is deliberately not a guess: a
    /// wrong cutoff doesn't fail loudly, it produces confident nonsense (see
    /// [`similarity_to_percent`]), so "no default" is the honest encoding of
    /// "not yet derived".
    ///
    /// SFace's 0.363 is published by OpenCV Zoo. ArcFace's has no published
    /// equivalent for this preprocessing, and deriving it needs a pair set
    /// large enough to be stable - two small sets disagreed by 0.119, far
    /// beyond their own ±0.033 within-set spread, so neither is usable.
    pub fn default_threshold(self) -> Option<f64> {
        match self {
            Backend::SFace => Some(DEFAULT_THRESHOLD),
            Backend::ArcFace => None,
        }
    }

    /// Lowercase name used by the CLI flag and in `--json` output.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::SFace => "sface",
            Backend::ArcFace => "arcface",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// ArcFace's canonical destination landmarks for a 112x112 aligned crop, from
/// the reference `face_preprocess.py` (base points, +8.0 on x for a 112-wide
/// crop).
///
/// The row order matches YuNet's own landmark order - right eye, left eye,
/// nose, right mouth corner, left mouth corner - so the two map straight onto
/// each other with no permutation.
///
/// Do not reorder these rows, and do not permute the source landmarks read out
/// of the detection row. Getting it wrong does not error and does not even look
/// broken: it silently inverts results. Measured on a same-person pair against
/// a different-person pair, the correct order separates them by +0.7315 while
/// a swapped one scores the *different* pair higher, at -0.0656. `tests/
/// arcface_alignment.rs` pins this.
pub const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Side length of the square crop ArcFace expects.
pub const ARCFACE_INPUT_SIZE: i32 = 112;

/// [`ARCFACE_DST`] as OpenCV points, in the same order.
pub fn arcface_dst_points() -> Vector<Point2f> {
    ARCFACE_DST
        .iter()
        .map(|[x, y]| Point2f::new(*x, *y))
        .collect()
}

/// Reads the 5 landmarks out of one YuNet detection row, in the detector's
/// native order.
///
/// A row is `[x, y, w, h, 5x(landmark x, landmark y), score]`, so the
/// landmarks are columns 4 through 13 taken in pairs, left to right. That
/// sequential read *is* the ordering contract with [`ARCFACE_DST`] - there is
/// no permutation step to get wrong, and adding one would break alignment
/// silently.
pub fn landmarks_from_row(face_row: &impl MatTraitConst) -> Result<Vector<Point2f>, CvError> {
    let mut points = Vector::<Point2f>::new();
    for i in 0..5 {
        let x = *face_row.at_2d::<f32>(0, 4 + i * 2)?;
        let y = *face_row.at_2d::<f32>(0, 5 + i * 2)?;
        points.push(Point2f::new(x, y));
    }
    Ok(points)
}

/// The 2x3 similarity transform mapping `src` landmarks onto [`ARCFACE_DST`].
///
/// RANSAC with a deliberately huge reprojection threshold keeps all five
/// points as inliers, which makes this a least-squares fit over the whole set
/// rather than a robust fit free to discard one. With only five points there
/// is nothing to gain by rejecting a landmark, and silently dropping one would
/// shift the alignment.
pub fn arcface_transform(src: &Vector<Point2f>) -> Result<Mat, CvError> {
    calib3d::estimate_affine_partial_2d(
        src,
        &arcface_dst_points(),
        &mut cv_core::no_array(),
        calib3d::RANSAC,
        1000.0,
        2000,
        0.99,
        10,
    )
}

/// Warps `image` so the face at `landmarks` lands on ArcFace's canonical
/// 112x112 layout.
///
/// Takes landmarks rather than a detection row so a caller can supply a
/// deliberately wrong order and measure the damage - see
/// `examples/ordering_check.rs`, which is how the ordering claim was checked
/// against real photographs rather than argued from the constants.
pub fn arcface_align(image: &Mat, landmarks: &Vector<Point2f>) -> Result<Mat, CvError> {
    let transform = arcface_transform(landmarks)?;
    let mut aligned = Mat::default();
    imgproc::warp_affine(
        image,
        &mut aligned,
        &transform,
        Size::new(ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        imgproc::INTER_LINEAR,
        cv_core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(aligned)
}

/// Runs an aligned 112x112 crop through the ArcFace graph and L2-normalises
/// the result, so a dot product against another embedding is a cosine.
///
/// ArcFace takes the crop as raw 0-255 values: no rescaling and no mean
/// subtraction, but with the channels swapped, since OpenCV decodes to BGR and
/// the model was trained on RGB.
pub fn arcface_features(net: &mut dnn::Net, aligned: &Mat) -> Result<Mat, CvError> {
    let blob = dnn::blob_from_image(
        aligned,
        1.0,
        Size::new(ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        Scalar::default(),
        true,
        false,
        CV_32F,
    )?;
    net.set_input(&blob, "", 1.0, Scalar::default())?;
    let raw = net.forward_single("")?;

    // The graph's output is unnormalised, so callers could not treat a dot
    // product as a cosine without this step.
    let mut feature = Mat::default();
    cv_core::normalize(
        &raw,
        &mut feature,
        1.0,
        0.0,
        NORM_L2,
        -1,
        &cv_core::no_array(),
    )?;
    Ok(feature)
}

/// YuNet detection confidence at/above which a candidate counts as a face.
///
/// OpenCV's own constructor default is 0.9, which is noticeably too strict for
/// ordinary photos: measured against a 64-image set of real-world photographs,
/// 0.9 found a face in only 41 of them (64%), 0.8 in 48 (75%), and 0.7 in 59
/// (92%). Since a missed detection is a hard "no face detected" failure while a
/// spurious one merely adds a low-similarity row, 0.7 is the better default.
///
/// Going lower isn't free: at 0.6 the detector starts firing on non-face
/// imagery and same/different separation begins to degrade, so 0.7 is about
/// where the curve turns.
pub const DEFAULT_DETECTION_CONFIDENCE: f32 = 0.7;

/// A face embedding: a single row of floats - 128 wide for SFace, 512 for
/// ArcFace.
pub type FaceEncoding = Mat;

/// How many numbers make up one face embedding.
///
/// This is the count of values actually compared between two faces - 128 for
/// SFace, 512 for ArcFace. It is read back from the embedding rather than
/// hardcoded, so it stays truthful if a different recognition model is passed
/// in. Note this is unrelated to the detector's 5 facial landmarks, which are
/// used only to align a face before embedding and are never compared.
pub fn embedding_dimensions(encoding: &FaceEncoding) -> i32 {
    encoding.cols()
}

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

/// The loaded recognition model, in whichever shape its backend needs.
enum Recognizer {
    /// SFace ships its own alignment and similarity, so it stays a
    /// `FaceRecognizerSF` rather than a bare graph.
    SFace(opencv::core::Ptr<FaceRecognizerSF>),
    /// ArcFace is a plain ONNX graph: alignment, preprocessing and
    /// normalisation are all this crate's responsibility.
    ArcFace(dnn::Net),
}

/// Loads OpenCV's YuNet detector and a recognition model once and reuses them
/// across many comparisons.
pub struct FaceComparer {
    detector: opencv::core::Ptr<FaceDetectorYN>,
    recognizer: Recognizer,
    backend: Backend,
}

impl FaceComparer {
    /// `detector_model` is a path to OpenCV Zoo's
    /// `face_detection_yunet_2023mar.onnx`. `recognizer_model` is the
    /// embedding model, which must match `backend`:
    /// `face_recognition_sface_2021dec.onnx` for [`Backend::SFace`],
    /// `arcfaceresnet100-11-int8.onnx` for [`Backend::ArcFace`].
    /// `detection_confidence` is YuNet's score threshold - see
    /// [`DEFAULT_DETECTION_CONFIDENCE`].
    pub fn new(
        detector_model: impl AsRef<Path>,
        recognizer_model: impl AsRef<Path>,
        detection_confidence: f32,
        backend: Backend,
    ) -> Result<Self, FacecompError> {
        let detector_model = path_to_str(detector_model.as_ref())?;
        let recognizer_model = path_to_str(recognizer_model.as_ref())?;

        // The real input size is set per-image in `detect`; this initial
        // size is just a placeholder required by the constructor.
        let detector = FaceDetectorYN::create(
            detector_model,
            "",
            Size::new(320, 320),
            detection_confidence,
            0.3,
            5000,
            0,
            0,
        )
        .map_err(|e| FacecompError::Model(e.to_string()))?;

        let recognizer = match backend {
            Backend::SFace => Recognizer::SFace(
                FaceRecognizerSF::create(recognizer_model, "", 0, 0)
                    .map_err(|e| FacecompError::Model(e.to_string()))?,
            ),
            Backend::ArcFace => Recognizer::ArcFace(
                dnn::read_net_from_onnx(recognizer_model)
                    .map_err(|e| FacecompError::Model(e.to_string()))?,
            ),
        };

        Ok(Self {
            detector,
            recognizer,
            backend,
        })
    }

    /// Which recognition model this comparer was built with.
    pub fn backend(&self) -> Backend {
        self.backend
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

    /// Compares two embeddings by cosine similarity.
    pub fn compare(
        &self,
        a: &FaceEncoding,
        b: &FaceEncoding,
        threshold: f64,
    ) -> Result<Comparison, FacecompError> {
        let similarity = match &self.recognizer {
            Recognizer::SFace(recognizer) => recognizer
                .match_(a, b, FaceRecognizerSF_FR_COSINE)
                .map_err(|e| FacecompError::Model(e.to_string()))?,
            // ArcFace embeddings leave `encode_row` L2-normalised, so their dot
            // product *is* the cosine of the angle between them - the same
            // quantity SFace's FR_COSINE returns, on the same 0..1 scale.
            Recognizer::ArcFace(_) => a
                .dot(b)
                .map_err(|e| FacecompError::Model(e.to_string()))?,
        };
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

        match &mut self.recognizer {
            Recognizer::SFace(recognizer) => {
                let mut aligned = Mat::default();
                recognizer
                    .align_crop(image, &face_row, &mut aligned)
                    .map_err(|e| cv_image_err(path, e))?;

                let mut feature = Mat::default();
                recognizer
                    .feature(&aligned, &mut feature)
                    .map_err(|e| cv_image_err(path, e))?;
                feature.try_clone().map_err(|e| cv_image_err(path, e))
            }
            Recognizer::ArcFace(net) => {
                let landmarks = landmarks_from_row(&face_row).map_err(|e| cv_image_err(path, e))?;
                let aligned =
                    arcface_align(image, &landmarks).map_err(|e| cv_image_err(path, e))?;
                arcface_features(net, &aligned).map_err(|e| cv_image_err(path, e))
            }
        }
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
///
/// `threshold` must lie in `(0.0, 1.0)`. At exactly 1.0 the divisor below is
/// zero and above it the scale inverts, so out-of-range values don't fail
/// loudly - they quietly return 0% for an identical face, or 100% for a
/// stranger. The CLI rejects them before they reach here; callers using this
/// crate directly need to do the same.
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
