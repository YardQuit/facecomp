use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;
use unicode_width::UnicodeWidthStr;

use facecomp::{
    confidence_label, embedding_dimensions, Backend, FaceComparer, DEFAULT_DETECTION_CONFIDENCE,
};

/// Compare a master photo against one or more other photos and report a
/// percentage match plus a qualitative confidence label for each.
#[derive(Parser)]
#[command(
    name = "facecomp",
    version,
    author = "Michael A Jones <yardquit@pm.me>",
    about = "Compare a master photo against one or more other photos and report a percentage match plus a confidence label for each",
    help_template = "{before-help}{name}({version}) Face Comparison\nCopyright (C) 2026 {author}\nLicensed under the GNU General Public License v3.0 (GPL-3.0-or-later);\nsee the LICENSE file distributed with this program for the full text.\n\n{about}\n{usage-heading} {usage}\n\n{all-args}{after-help}\n",
    after_help = "CONFIDENCE LABELS:\n    Almost certain    95-99%\n    Very likely       80-95%\n    Likely            55-80%\n    Even chance       45-55%\n    Unlikely          20-45%\n    Very unlikely      5-20%\n    Almost no chance   1-5%\n\n    Publisher: Office of the Director of National Intelligence (ODNI)\n\nMULTIPLE FACES:\n    If a --slave photo has more than one person in it, every face found is\n    compared against the master and the best match is reported. The `faces`\n    column (or `faces_detected` in --json) shows how many were found.\n\nHOW FACES ARE COMPARED:\n    Each face is reduced to an embedding - a fixed list of numbers describing\n    it - and two faces are compared by the cosine similarity between their\n    embeddings. The shipped SFace model produces 128 numbers per face; the\n    exact count for the model in use is reported as `embedding` in the output\n    (`embedding_dimensions` in --json).\n\n    The detector also finds 5 facial landmarks (eyes, nose, mouth corners),\n    but those are used only to align a face before embedding it. They are not\n    themselves compared, so they don't add to the numbers above.\n\nCHOOSING --detection-confidence:\n    This decides which photos yield a face at all; it is not what governs how\n    accurate a comparison is (that is the model, and --threshold). Measured\n    over 64 real-world photographs:\n\n        0.9    face found in 41 (64%)     no false detections\n        0.8    face found in 48 (75%)     no false detections\n        0.7    face found in 59 (92%)     one, on a photo of dogs   [default]\n        0.6    face found in 63 (98%)     two\n        0.5    face found in 64 (100%)    two\n\n    Use 0.8 when every result needs to be trustworthy: it never picked up a\n    non-face, and still finds more faces than 0.9 - there is no reason to run\n    0.9 at all. Keep the 0.7 default when you would rather not silently skip\n    photos; a spurious detection only adds a low-similarity row (the dog photo\n    scored 14%, \"Very unlikely\"). Below 0.6 the detector starts firing on\n    genuinely non-face imagery.\n\n    A marginal detection also gives sloppier landmarks, so the face is aligned\n    less precisely before embedding - a further reason to re-check borderline\n    results at 0.8."
)]
struct Args {
    /// The reference photo every other photo is compared against.
    #[arg(long)]
    master: PathBuf,

    /// A photo to compare against the master; repeat or pass multiple
    /// values to compare several. Each may be a literal path or a glob
    /// pattern (e.g. "photos/*.png") - pass a pattern in quotes if your
    /// shell doesn't expand wildcards itself.
    #[arg(long = "slave", required = true, num_args = 1..)]
    slaves: Vec<String>,

    /// Path to OpenCV Zoo's face_detection_yunet_2023mar.onnx.
    #[arg(long, env = "FACECOMP_DETECTOR_MODEL")]
    detector_model: PathBuf,

    /// Which embedding model to use: "sface" (128 numbers per face) or
    /// "arcface" (512). --encoder-model must be the matching weights.
    #[arg(long, default_value = "sface", value_parser = parse_backend)]
    backend: Backend,

    /// Path to the embedding model matching --backend. Defaults to
    /// $FACECOMP_ENCODER_MODEL for sface, $FACECOMP_ARCFACE_MODEL for arcface.
    #[arg(long)]
    encoder_model: Option<PathBuf>,

    /// Cosine similarity at/above which two faces count as the same person.
    /// Defaults to 0.363 for sface; has no default for arcface, which has no
    /// trustworthy value derived yet, so pass one explicitly.
    #[arg(long, value_parser = parse_threshold)]
    threshold: Option<f64>,

    /// Report only the N closest-matching photos rather than every one
    /// compared. Useful when --slave expands to a large directory.
    #[arg(long, value_parser = parse_max)]
    max: Option<usize>,

    /// Detector confidence at/above which a candidate counts as a face. Lower
    /// it to find faces in difficult photos; raise it if non-faces are picked up.
    #[arg(long, env = "FACECOMP_DETECTION_CONFIDENCE",
          default_value_t = DEFAULT_DETECTION_CONFIDENCE,
          value_parser = parse_detection_confidence)]
    detection_confidence: f32,

    /// Emit machine-readable JSON instead of a text table.
    #[arg(long)]
    json: bool,
}

/// Width of the `photo` column in the text table.
const PHOTO_COLUMN: usize = 30;

/// Pads `s` to `width` terminal columns, not to `width` characters.
///
/// `{:<30}` counts characters, which is the same thing only for narrow
/// scripts. CJK and other East Asian Wide characters occupy two columns each,
/// so a filename like `日本語.jpg` was padded as though it were half as wide
/// as it prints, and every column after it drifted left. Over-long names are
/// left to overflow, matching the previous behaviour.
fn pad_to_width(s: &str, width: usize) -> String {
    let printed = UnicodeWidthStr::width(s);
    if printed >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - printed))
    }
}

/// Rejects thresholds that make the match-percent scale meaningless.
///
/// Cosine similarity tops out at 1.0, so a threshold at or above it can never
/// be met - and worse, it doesn't merely fail, it misreports. The scale is
/// centred by dividing by `1 - (2*threshold - 1)`, which is zero at exactly
/// 1.0 and negative beyond it: a threshold of 1.0 reported an identical face
/// as "0.0% Almost no chance", and 1.5 reported a stranger as "100% Almost
/// certain". Refusing the value up front is the only honest option.
fn parse_threshold(s: &str) -> Result<f64, String> {
    let value: f64 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a number"))?;
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err(format!(
            "must be greater than 0 and less than 1, got {value} \
             (cosine similarity never exceeds 1.0)"
        ))
    }
}

/// Rejects detector confidences outside the range YuNet scores can occupy.
///
/// The score is a probability, so anything at or below 0 accepts every
/// candidate the network proposes - 0 turned a single portrait into 1543
/// "faces" and reported a best match against the noise.
fn parse_detection_confidence(s: &str) -> Result<f32, String> {
    let value: f32 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a number"))?;
    if value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err(format!("must be greater than 0 and at most 1, got {value}"))
    }
}

/// Parses `--backend`. Kept here rather than as a `clap::ValueEnum` on
/// [`Backend`] so the library doesn't take a dependency on clap.
fn parse_backend(s: &str) -> Result<Backend, String> {
    match s.to_ascii_lowercase().as_str() {
        "sface" => Ok(Backend::SFace),
        "arcface" => Ok(Backend::ArcFace),
        other => Err(format!(
            "unknown backend `{other}` (expected `sface` or `arcface`)"
        )),
    }
}

/// Rejects `--max 0`, which would silently report nothing at all rather than
/// erroring - the same class of quietly-wrong output `--threshold 1.0` gave.
fn parse_max(s: &str) -> Result<usize, String> {
    let value: usize = s
        .parse()
        .map_err(|_| format!("`{s}` is not a whole number"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("must be at least 1".to_string())
    }
}

/// Resolves which weights file to load, given the backend and what the user
/// supplied.
///
/// `--encoder-model` wins if present. Otherwise each backend reads its own
/// environment variable, so the AppImage can export both bundled models up
/// front and let `--backend` decide between them at run time.
fn resolve_encoder_model(
    explicit: Option<PathBuf>,
    backend: Backend,
) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    let var = match backend {
        Backend::SFace => "FACECOMP_ENCODER_MODEL",
        Backend::ArcFace => "FACECOMP_ARCFACE_MODEL",
    };
    std::env::var_os(var).map(PathBuf::from).ok_or_else(|| {
        format!("no model for --backend {backend}: pass --encoder-model or set ${var}")
    })
}

/// Resolves the same/different cutoff, refusing to invent one.
///
/// SFace has a published threshold to fall back on; ArcFace does not, and
/// guessing would produce confident nonsense rather than an error. Requiring
/// the flag is the honest failure mode until a value is derived on a pair set
/// big enough to trust.
fn resolve_threshold(explicit: Option<f64>, backend: Backend) -> Result<f64, String> {
    explicit
        .or_else(|| backend.default_threshold())
        .ok_or_else(|| {
            format!(
                "--backend {backend} has no default --threshold yet, so one must be given \
                 explicitly.\n       No trustworthy value has been derived for it: two small \
                 pair sets disagreed by 0.119, far beyond their own ±0.033 spread, so neither \
                 is usable as a default."
            )
        })
}

#[derive(Serialize)]
struct MasterComparison {
    photo: PathBuf,
    faces_detected: usize,
    similarity: f64,
    match_percent: f64,
    confidence: &'static str,
}

#[derive(Serialize)]
struct Report {
    master: PathBuf,
    backend: &'static str,
    threshold: f64,
    embedding_dimensions: i32,
    results: Vec<MasterComparison>,
    errors: Vec<String>,
}

/// Expands each pattern that looks like a glob (contains `*`, `?`, or `[`)
/// via the filesystem; everything else is taken as a literal path. The
/// master photo is excluded in case a glob happens to sweep it up.
fn expand_slaves(master: &Path, patterns: &[String]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for pattern in patterns {
        if pattern.contains(['*', '?', '[']) {
            match glob::glob(pattern) {
                Ok(entries) => paths.extend(entries.flatten()),
                Err(e) => eprintln!("warning: invalid glob pattern {pattern:?}: {e}"),
            }
        } else {
            paths.push(PathBuf::from(pattern));
        }
    }

    let master_canon = master.canonicalize().ok();
    paths.retain(|p| match (&master_canon, p.canonicalize()) {
        (Some(m), Ok(pc)) => &pc != m,
        _ => p != master,
    });
    paths.sort();
    paths.dedup();
    paths
}

fn main() -> ExitCode {
    let args = Args::parse();

    let encoder_model = match resolve_encoder_model(args.encoder_model, args.backend) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("facecomp: {e}");
            return ExitCode::FAILURE;
        }
    };

    let threshold = match resolve_threshold(args.threshold, args.backend) {
        Ok(threshold) => threshold,
        Err(e) => {
            eprintln!("facecomp: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut comparer = match FaceComparer::new(
        &args.detector_model,
        &encoder_model,
        args.detection_confidence,
        args.backend,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("facecomp: {e}");
            return ExitCode::FAILURE;
        }
    };

    let slaves = expand_slaves(&args.master, &args.slaves);
    if slaves.is_empty() {
        eprintln!(
            "facecomp: no slave photos found (after expanding glob patterns and excluding the master)"
        );
        return ExitCode::FAILURE;
    }

    let mut errors = Vec::new();

    let master_encoding = match comparer.encode_face(&args.master) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("facecomp: master photo: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut results = Vec::new();
    for slave in &slaves {
        match comparer.encode_all_faces(slave) {
            Ok(encodings) => {
                // A slave photo may contain more than one person; compare the
                // master against every face found and keep the best match,
                // rather than assuming there's only one face in frame.
                let comparisons: Result<Vec<_>, _> = encodings
                    .iter()
                    .map(|encoding| comparer.compare(&master_encoding, encoding, threshold))
                    .collect();
                match comparisons {
                    Ok(comparisons) => {
                        let best = comparisons
                            .into_iter()
                            .max_by(|a, b| a.match_percent.total_cmp(&b.match_percent))
                            .expect("encode_all_faces never returns an empty Vec on success");
                        results.push(MasterComparison {
                            photo: slave.clone(),
                            faces_detected: encodings.len(),
                            similarity: best.similarity,
                            match_percent: best.match_percent,
                            confidence: confidence_label(best.match_percent),
                        });
                    }
                    Err(e) => errors.push(format!("{}: {e}", slave.display())),
                }
            }
            Err(e) => errors.push(format!("{}: {e}", slave.display())),
        }
    }

    // Asking for the closest N matches only means something once they're
    // ranked, so --max sorts before truncating. Without it the photos stay in
    // the order they were given, which is what every previous version did.
    if let Some(max) = args.max {
        results.sort_by(|a, b| b.match_percent.total_cmp(&a.match_percent));
        results.truncate(max);
    }

    let embedding_dims = embedding_dimensions(&master_encoding);

    if args.json {
        let report = Report {
            master: args.master,
            backend: args.backend.as_str(),
            threshold,
            embedding_dimensions: embedding_dims,
            results,
            errors: errors.clone(),
        };
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        for err in &errors {
            eprintln!("warning: {err}");
        }
        println!("master: {}", args.master.display());
        println!("backend: {}", args.backend);
        println!("embedding: {embedding_dims} dimensions per face\n");
        println!(
            "{} {:>6} {:>10} {:>8}  confidence",
            pad_to_width("photo", PHOTO_COLUMN),
            "faces",
            "similarity",
            "match %"
        );
        for r in &results {
            println!(
                "{} {:>6} {:>10.4} {:>7.1}%  {}",
                pad_to_width(&r.photo.display().to_string(), PHOTO_COLUMN),
                r.faces_detected,
                r.similarity,
                r.match_percent,
                r.confidence,
            );
        }
    }

    if !errors.is_empty() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
