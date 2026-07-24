# facecomp

Compare a master photo against one or more other photos and get a
distance-based percentage match plus a qualitative confidence label
for each — usable as a standalone command-line tool, or from Emacs.

## How it works

- `facecomp` (Rust) detects every face in the master photo and in each
  other photo using OpenCV's YuNet detector, computes a 128-dimension
  face embedding for each using OpenCV's SFace, and reports the cosine
  similarity between the master's embedding and every other photo's.
  Similarity is mapped onto a 0-100% "match" heuristic (see
  [Interpreting the percentage](#interpreting-the-percentage)) and
  onto a qualitative confidence label (see
  [Confidence labels](#confidence-labels)).
- `emacs/facecomp.el` is a thin frontend: it shells out to the
  `facecomp` binary and renders the JSON result in an Emacs buffer.
  The Rust binary does not depend on Emacs in any way.

## Building

### System requirements

- Rust (stable) and Clang/libclang (needed by the `opencv` crate to
  generate bindings).
- OpenCV **4.10 or newer**, with the `objdetect` and `dnn` modules, plus
  their headers (`libopencv-dev` or equivalent).

  **Important:** OpenCV older than roughly 4.10 (including the
  `libopencv-dev` 4.6.0 that Ubuntu 24.04's own apt repos ship) cannot
  load the YuNet detector model at all — it fails at runtime with a
  DNN importer error (`Layer with requested id=-1 not found`), not at
  build time, so this is easy to miss until you actually run the
  binary. This is why `packaging/build-appimage.sh` and the AppImage CI
  workflow build OpenCV from source instead of using the distro
  package. If you build `facecomp` outside that pipeline, either build
  a recent OpenCV from source yourself, or install one from a source
  that ships a newer version than your distro's default repos.

### Build

```sh
cargo build --release
```

The resulting binary is at `target/release/facecomp`. Point
`PKG_CONFIG_PATH` at your OpenCV build's `lib/pkgconfig` directory
first if it isn't the one your system's `pkg-config` would find by
default.

## Model files

`facecomp` needs two of OpenCV Zoo's pretrained model files at
runtime. They are not checked into this repository (the recognition
model alone is ~37 MB) and must be downloaded separately:

- `face_detection_yunet_2023mar.onnx` — detects faces and 5-point
  landmarks.
- `face_recognition_sface_2021dec.onnx` — produces the 128-d face
  embedding.

Get both from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
repository, under `models/face_detection_yunet/` and
`models/face_recognition_sface/` respectively (they're stored via Git
LFS, so a plain file download from GitHub's UI works, but a shallow
`git clone` without LFS will only get you pointer files).

## Portable AppImage build

Since the compiled binary alone isn't enough to run on another
machine — it also needs a modern-enough OpenCV's shared libraries
(distro packages are typically too old, see above) and both model
files present — `packaging/build-appimage.sh` builds a minimal OpenCV
from source and bundles everything into one self-contained
`facecomp-x86_64.AppImage`:

```sh
./packaging/build-appimage.sh \
  /path/to/face_detection_yunet_2023mar.onnx \
  /path/to/face_recognition_sface_2021dec.onnx
```

This builds OpenCV (core/imgproc/imgcodecs/objdetect/dnn only, to keep
build time down — expect roughly 10 minutes on a few cores, cached
across runs in CI) and `facecomp` in release mode, fetches
`linuxdeploy` and `appimagetool` on first run (cached under
`packaging/.tools/`), and produces `facecomp-x86_64.AppImage` in the
repo root. The result:

- Runs on other Linux x86_64 machines without needing OpenCV, cmake,
  or a C++ compiler installed — the shared libraries this build
  produces are bundled alongside the binary and preferred over any
  system copies via `LD_LIBRARY_PATH`.
- `--master`/`--slave`/etc. work exactly as documented below —
  `--detector-model`/`--encoder-model` don't need to be passed since
  the AppImage's `AppRun` wrapper points them at the bundled model
  files automatically (still overridable via
  `FACECOMP_DETECTOR_MODEL`/`FACECOMP_ENCODER_MODEL` if you want to
  point at different ones).
- Works whether or not the target machine has FUSE: if `fusermount`
  isn't available to mount the AppImage, its runtime automatically
  falls back to self-extracting, no `--appimage-extract-and-run` flag
  needed.

Caveat: the produced binary requires a glibc version at least as new
as whatever machine you build it on (glibc is forward-compatible only)
— build on the oldest Linux you expect to target, not the newest.

## CLI usage

```sh
facecomp \
  --detector-model /path/to/face_detection_yunet_2023mar.onnx \
  --encoder-model  /path/to/face_recognition_sface_2021dec.onnx \
  --master master.jpg \
  --slave photo1.jpg photo2.jpg
```

`--slave` takes one or more photos (or repeat the flag). The master is
compared against each one (not against itself, and slaves are not
compared against each other):

```
master: master.jpg

photo                           faces  similarity  match %  confidence      same?
photo1.jpg                        1        0.7437    79.9%  Likely          yes
photo2.jpg                        1        0.0992    29.3%  Unlikely        no
```

If a slave photo has more than one person in it, `facecomp` compares
the master against every face detected and reports the best match —
the `faces` column shows how many faces were found, so a value above 1
tells you the result was picked from multiple candidates rather than
just the only face in frame.

Each `--slave` value can also be a glob pattern instead of a literal
path — useful when your shell doesn't expand wildcards itself, or
you'd rather not rely on shell expansion at all:

```sh
facecomp --master master.jpg --detector-model ... --encoder-model ... --slave "photos/*.png"
```

(Quote the pattern so your shell passes it through literally.) If a
glob happens to match the master photo itself, it's excluded
automatically.

Model paths can also come from environment variables instead of flags:
`FACECOMP_DETECTOR_MODEL` and `FACECOMP_ENCODER_MODEL`.

Other flags:

- `--threshold <f64>` — the cosine similarity at/above which two faces
  count as the same person (default `0.363`, the OpenCV Zoo-published
  recommendation for the SFace model).
- `--json` — emit a machine-readable JSON report instead of the table
  above (this is what `facecomp.el` uses).

Exit status is non-zero if any photo failed (typically: no face was
detected in it, or a glob matched nothing).

### Interpreting the percentage

The cosine similarity itself (1.0 = identical, lower = less alike, and
in principle as low as -1.0) is the only number OpenCV Zoo actually
calibrates — its documented same/different cutoff for the SFace model
is 0.363. The "match %" is a heuristic linear rescaling of that
similarity, chosen so the 0.363 cutoff lands at exactly 50%:

```
match% = clamp(100 * (similarity - (2*threshold - 1)) / (1 - (2*threshold - 1)), 0, 100)
```

It is not a calibrated probability of "same person" — treat it as a
convenient, threshold-centered readout of the underlying similarity,
not ground truth.

### Confidence labels

Alongside the raw percentage, each result also gets a qualitative
label from the intelligence-community "words of estimative
probability" yardstick (the same bands used in ICD 203):

> Publisher: Office of the Director of National Intelligence (ODNI)

| Label                        | Match %  |
|-------------------------------|---------|
| Almost certain / Nearly certain | 95-99%  |
| Very likely / Highly likely     | 80-95%  |
| Likely / probable               | 55-80%  |
| Even chance / roughly           | 45-55%  |
| Unlikely / improbable           | 20-45%  |
| Very unlikely / Highly unlikely | 5-20%   |
| Almost no chance / remote       | 1-5%    |

The bands overlap at their edges in the original yardstick; `facecomp`
resolves that with non-overlapping cutoffs (`>= 95`, `>= 80`, `>= 55`,
`>= 45`, `>= 20`, `>= 5`, else the bottom band) so every percentage
maps to exactly one label.

## Emacs usage

Load `emacs/facecomp.el` and configure it:

```elisp
(use-package facecomp
  :load-path "/path/to/facecomp/emacs"
  :custom
  (facecomp-detector-model "/path/to/face_detection_yunet_2023mar.onnx")
  (facecomp-encoder-model "/path/to/face_recognition_sface_2021dec.onnx")
  ;; Only needed if the `facecomp` binary isn't already on PATH:
  (facecomp-executable "/path/to/facecomp/target/release/facecomp"))
```

Then:

- `M-x facecomp-compare` prompts for a master photo, then either a
  glob pattern (e.g. `*.png`) or photos to compare against it picked
  one at a time.
- Or mark two or more files in Dired and run `M-x facecomp-compare` —
  the first marked file becomes the master, the rest are compared
  against it.

Results are shown in a `*facecomp*` buffer, one photo per entry, with
the match percentage and confidence label colored by confidence.

## License

GPL-3.0-or-later. See `LICENSE`.
