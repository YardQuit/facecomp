use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;

use facecomp::{confidence_label, FaceComparer, DEFAULT_THRESHOLD};

/// Compare a master photo against one or more other photos and report a
/// percentage match plus a qualitative confidence label for each.
#[derive(Parser)]
#[command(
    name = "facecomp",
    version,
    author = "Michael A Jones <yardquit@pm.me>",
    about = "Compare a master photo against one or more other photos and report a percentage match plus a confidence label for each",
    help_template = "{before-help}{name}({version}) Face Comparison\nCopyright (C) 2026 {author}\nLicensed under the GNU General Public License v3.0 (GPL-3.0-or-later);\nsee the LICENSE file distributed with this program for the full text.\n\n{about}\n{usage-heading} {usage}\n\n{all-args}{after-help}\n",
    after_help = "CONFIDENCE LABELS:\n    Almost certain    95-99%\n    Very likely       80-95%\n    Likely            55-80%\n    Even chance       45-55%\n    Unlikely          20-45%\n    Very unlikely      5-20%\n    Almost no chance   1-5%\n\n    Publisher: Office of the Director of National Intelligence (ODNI)\n\nMULTIPLE FACES:\n    If a --slave photo has more than one person in it, every face found is\n    compared against the master and the best match is reported. The `faces`\n    column (or `faces_detected` in --json) shows how many were found."
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

    /// Path to OpenCV Zoo's face_recognition_sface_2021dec.onnx.
    #[arg(long, env = "FACECOMP_ENCODER_MODEL")]
    encoder_model: PathBuf,

    /// Cosine similarity at/above which two faces count as the same person.
    #[arg(long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f64,

    /// Emit machine-readable JSON instead of a text table.
    #[arg(long)]
    json: bool,
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

    let mut comparer = match FaceComparer::new(&args.detector_model, &args.encoder_model) {
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
                    .map(|encoding| comparer.compare(&master_encoding, encoding, args.threshold))
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
            "{:<30} {:>6} {:>10} {:>8}  confidence",
            "photo", "faces", "similarity", "match %"
        );
        for r in &results {
            println!(
                "{:<30} {:>6} {:>10.4} {:>7.1}%  {}",
                r.photo.display(),
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
