# facecomp

Compare a master photo against one or more other photos and get a
distance-based percentage match plus a qualitative confidence label
for each — usable as a standalone command-line tool, or from Emacs.

## How it works

- `facecomp` (Rust) detects the most prominent face in the master
  photo and in each other photo, computes a 128-dimension face
  embedding for each using dlib, and reports the Euclidean distance
  between the master's embedding and every other photo's. Distance is
  mapped onto a 0-100% "match" heuristic (see
  [Interpreting the percentage](#interpreting-the-percentage)) and
  onto a qualitative confidence label (see
  [Confidence labels](#confidence-labels)).
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
- If your CMake is version 4.x: dlib 19.24's bundled `CMakeLists.txt`
  requires a CMake version older than 3.5, which CMake 4.x refuses to
  configure. `.cargo/config.toml` in this repo already sets
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` to work around it, so this should
  be transparent when building from a checkout of this repo.
- If your toolchain is new enough that dlib's `'uint8_t' was not
  declared in this scope` compile errors show up (recent GCC/libstdc++
  no longer transitively pulls in `<cstdint>` the way dlib 19.24's
  headers assume): also already handled, via `CXXFLAGS=-include cstdint`
  in `.cargo/config.toml`.

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
  --master master.jpg \
  --slave photo1.jpg photo2.jpg
```

`--slave` takes one or more photos (or repeat the flag). The master is
compared against each one (not against itself, and slaves are not
compared against each other):

```
master: master.jpg

photo                            distance  match %  confidence      same?
photo1.jpg                         0.1991    83.4%  Very likely     yes
photo2.jpg                         0.6965    42.0%  Unlikely        no
```

Each `--slave` value can also be a glob pattern instead of a literal
path — useful when your shell doesn't expand wildcards itself, or
you'd rather not rely on shell expansion at all:

```sh
facecomp --master master.jpg --landmark-model ... --encoder-model ... --slave "photos/*.png"
```

(Quote the pattern so your shell passes it through literally.) If a
glob happens to match the master photo itself, it's excluded
automatically.

Model paths can also come from environment variables instead of flags:
`FACECOMP_LANDMARK_MODEL` and `FACECOMP_ENCODER_MODEL`.

Other flags:

- `--threshold <f64>` — the embedding distance at/below which two faces
  count as the same person (default `0.6`, dlib's own published
  recommendation).
- `--json` — emit a machine-readable JSON report instead of the table
  above (this is what `facecomp.el` uses).

Exit status is non-zero if any photo failed (typically: no face was
detected in it, or a glob matched nothing).

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

### Confidence labels

Alongside the raw percentage, each result also gets a qualitative
label from the intelligence-community "words of estimative
probability" yardstick (the same bands used in ICD 203):

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
  (facecomp-landmark-model "/path/to/shape_predictor_68_face_landmarks.dat")
  (facecomp-encoder-model "/path/to/dlib_face_recognition_resnet_model_v1.dat")
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
