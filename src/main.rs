use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;

use facecomp::{compare, confidence_label, FaceComparer, DEFAULT_THRESHOLD};

/// Compare a master photo against one or more other photos and report a
/// percentage match plus a qualitative confidence label for each.
#[derive(Parser)]
#[command(name = "facecomp", version, about)]
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
struct MasterComparison {
    photo: PathBuf,
    distance: f64,
    match_percent: f64,
    confidence: &'static str,
    same_person: bool,
}

#[derive(Serialize)]
struct Report {
    master: PathBuf,
    threshold: f64,
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

    let comparer = match FaceComparer::new(&args.landmark_model, &args.encoder_model) {
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
        match comparer.encode_face(slave) {
            Ok(encoding) => {
                let result = compare(&master_encoding, &encoding, args.threshold);
                results.push(MasterComparison {
                    photo: slave.clone(),
                    distance: result.distance,
                    match_percent: result.match_percent,
                    confidence: confidence_label(result.match_percent),
                    same_person: result.same_person,
                });
            }
            Err(e) => errors.push(format!("{}: {e}", slave.display())),
        }
    }

    if args.json {
        let report = Report {
            master: args.master,
            threshold: args.threshold,
            results,
            errors: errors.clone(),
        };
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        for err in &errors {
            eprintln!("warning: {err}");
        }
        println!("master: {}\n", args.master.display());
        println!(
            "{:<30} {:>10} {:>8}  {:<16}same?",
            "photo", "distance", "match %", "confidence"
        );
        for r in &results {
            println!(
                "{:<30} {:>10.4} {:>7.1}%  {:<16}{}",
                r.photo.display(),
                r.distance,
                r.match_percent,
                r.confidence,
                if r.same_person { "yes" } else { "no" }
            );
        }
    }

    if !errors.is_empty() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
