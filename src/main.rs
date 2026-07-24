use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;

use facecomp::{compare, FaceComparer, DEFAULT_THRESHOLD};

/// Compare faces across two or more photos and report a percentage match per pair.
#[derive(Parser)]
#[command(name = "facecomp", version, about)]
struct Args {
    /// Two or more image files to compare; every pair is compared.
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,

    /// Path to dlib's shape_predictor_68_face_landmarks.dat.
    #[arg(long, env = "FACECOMP_LANDMARK_MODEL")]
    landmark_model: PathBuf,

    /// Path to dlib's dlib_face_recognition_resnet_model_v1.dat.
    #[arg(long, env = "FACECOMP_ENCODER_MODEL")]
    encoder_model: PathBuf,

    /// Euclidean distance at/below which two faces count as the same person.
    #[arg(long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f64,

    /// Emit machine-readable JSON instead of a text table.
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct PairResult {
    image_a: PathBuf,
    image_b: PathBuf,
    distance: f64,
    match_percent: f64,
    same_person: bool,
}

#[derive(Serialize)]
struct Report {
    threshold: f64,
    pairs: Vec<PairResult>,
    errors: Vec<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let comparer = match FaceComparer::new(&args.landmark_model, &args.encoder_model) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("facecomp: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut encodings = Vec::with_capacity(args.images.len());
    let mut errors = Vec::new();

    for path in &args.images {
        match comparer.encode_face(path) {
            Ok(encoding) => encodings.push(Some(encoding)),
            Err(e) => {
                errors.push(format!("{}: {e}", path.display()));
                encodings.push(None);
            }
        }
    }

    let mut pairs = Vec::new();
    for i in 0..args.images.len() {
        for j in (i + 1)..args.images.len() {
            if let (Some(a), Some(b)) = (&encodings[i], &encodings[j]) {
                let result = compare(a, b, args.threshold);
                pairs.push(PairResult {
                    image_a: args.images[i].clone(),
                    image_b: args.images[j].clone(),
                    distance: result.distance,
                    match_percent: result.match_percent,
                    same_person: result.same_person,
                });
            }
        }
    }

    if args.json {
        let report = Report {
            threshold: args.threshold,
            pairs,
            errors: errors.clone(),
        };
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        for err in &errors {
            eprintln!("warning: {err}");
        }
        println!(
            "{:<30} {:<30} {:>10} {:>8}  same?",
            "image A", "image B", "distance", "match %"
        );
        for p in &pairs {
            println!(
                "{:<30} {:<30} {:>10.4} {:>7.1}%  {}",
                p.image_a.display(),
                p.image_b.display(),
                p.distance,
                p.match_percent,
                if p.same_person { "yes" } else { "no" }
            );
        }
    }

    if !errors.is_empty() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
