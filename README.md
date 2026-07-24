# facecomp

Compare faces across two or more photos and get a distance-based
percentage match per pair — usable as a standalone command-line tool,
or from Emacs.

## How it works

- `facecomp` (Rust) detects the most prominent face in each photo,
  computes a 128-dimension face embedding for it using dlib, and
  reports the Euclidean distance between every pair of embeddings.
  Distance is mapped onto a 0-100% "match" heuristic (see
  [Interpreting the percentage](#interpreting-the-percentage)).
- `emacs/facecomp.el` is a thin frontend: it shells out to the
  `facecomp` binary and renders the JSON result in an Emacs buffer.
  The Rust binary does not depend on Emacs in any way.

## Building

### System requirements

- Rust (stable).
- `cmake`, a C++ compiler, and BLAS/LAPACK development headers, needed
  to compile dlib itself. On Debian/Ubuntu:

  ```sh
  sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev
  ```

- Roughly 200 MB of disk and a few minutes of CPU time — the first
  build compiles dlib from source (via the `dlib-face-recognition`
  crate's `build-native` feature).

### Build

```sh
cargo build --release
```

The resulting binary is at `target/release/facecomp`.

## Model files

`facecomp` needs two of dlib's pretrained model files at runtime. They
are not checked into this repository (they're tens of MB each) and
must be downloaded separately:

- `shape_predictor_68_face_landmarks.dat` — locates facial landmarks.
- `dlib_face_recognition_resnet_model_v1.dat` — produces the 128-d
  face embedding.

Get both from dlib's own site: <http://dlib.net/files/>, as
`.dat.bz2` archives that need decompressing (`bunzip2`). On
Debian/Ubuntu, the `libdlib-data` package also ships the landmark file
at `/usr/share/dlib/shape_predictor_68_face_landmarks.dat`, but the
face-recognition ResNet model still has to come from dlib.net.

## CLI usage

```sh
facecomp \
  --landmark-model /path/to/shape_predictor_68_face_landmarks.dat \
  --encoder-model  /path/to/dlib_face_recognition_resnet_model_v1.dat \
  photo1.jpg photo2.jpg photo3.jpg
```

Every pair among the given images is compared:

```
image A                        image B                          distance  match %  same?
photo1.jpg                     photo2.jpg                         0.1991    83.4%  yes
photo1.jpg                     photo3.jpg                         0.6965    42.0%  no
photo2.jpg                     photo3.jpg                         0.7143    40.5%  no
```

Model paths can also come from environment variables instead of flags:
`FACECOMP_LANDMARK_MODEL` and `FACECOMP_ENCODER_MODEL`.

Other flags:

- `--threshold <f64>` — the embedding distance at/below which two faces
  count as the same person (default `0.6`, dlib's own published
  recommendation).
- `--json` — emit a machine-readable JSON report instead of the table
  above (this is what `facecomp.el` uses).

Exit status is non-zero if any image failed (typically: no face was
detected in it).

### Interpreting the percentage

The embedding distance itself (0 = identical, larger = less alike) is
the only number dlib actually calibrates — its documented same/different
cutoff is 0.6. The "match %" is a heuristic linear rescaling of that
distance, chosen so the 0.6 cutoff lands at exactly 50%:

```
match% = clamp(100 * (1 - distance / (2 * threshold)), 0, 100)
```

It is not a calibrated probability of "same person" — treat it as a
convenient, threshold-centered readout of the underlying distance, not
ground truth.

## Emacs usage

Load `emacs/facecomp.el` and configure it:

```elisp
(use-package facecomp
  :load-path "/path/to/facecomp/emacs"
  :custom
  (facecomp-landmark-model "/path/to/shape_predictor_68_face_landmarks.dat")
  (facecomp-encoder-model "/path/to/dlib_face_recognition_resnet_model_v1.dat")
  ;; Only needed if the `facecomp` binary isn't already on PATH:
  (facecomp-executable "/path/to/facecomp/target/release/facecomp"))
```

Then:

- `M-x facecomp-compare` prompts for two or more image files.
- Or mark two or more files in Dired and run `M-x facecomp-compare` —
  it compares the marked files instead of prompting.

Results are shown in a `*facecomp*` buffer, one pair per entry, with
the match percentage colored by confidence.

## License

GPL-3.0-or-later. See `LICENSE`.
