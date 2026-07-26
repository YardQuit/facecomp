# facecomp

Compare a master photo against one or more other photos and get a
distance-based percentage match plus a qualitative confidence label
for each — usable as a standalone command-line tool, or from Emacs.

## How it works

- `facecomp` (Rust) detects every face in the master photo and in each
  other photo using OpenCV's YuNet detector, computes a face embedding
  for each using the model chosen by `--backend` (SFace by default, or
  ArcFace — see [Choosing a backend](#choosing-a-backend)), and reports
  the cosine similarity between the master's embedding and every other
  photo's.
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
- OpenCV **4.10 or newer**, with the `objdetect`, `dnn`, `imgproc` and
  `calib3d` modules, plus their headers (`libopencv-dev` or
  equivalent). `calib3d` supplies `estimateAffinePartial2D`, which
  aligns faces for the ArcFace backend; a build without it fails at
  compile time on missing headers.

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

`facecomp` needs pretrained model files at runtime. They are not
checked into this repository (the recognition models are ~37 MB and
~66 MB) and must be downloaded separately:

- `face_detection_yunet_2023mar.onnx` — detects faces and 5-point
  landmarks. Always required.
- `face_recognition_sface_2021dec.onnx` — produces the 128-d face
  embedding. Required for `--backend sface`, the default.
- `arcfaceresnet100-11-int8.onnx` — produces the 512-d face embedding.
  Required only for `--backend arcface`.

The first two come from the [OpenCV
Zoo](https://github.com/opencv/opencv_zoo) repository, under
`models/face_detection_yunet/` and `models/face_recognition_sface/`
respectively. ArcFace comes from the [ONNX Model
Zoo](https://github.com/onnx/models), under
`validated/vision/body_analysis/arcface/model/`. All are stored via
Git LFS, so a plain file download from GitHub's UI works, but a
shallow `git clone` without LFS will only get you pointer files.

Use the `int8` ArcFace build rather than `fp32`: the latter is 261 MB
for roughly 5% more separation, which isn't worth quadrupling the
download or the AppImage.

### Choosing a backend

|                | `sface` (default) | `arcface`      |
|----------------|-------------------|----------------|
| Embedding size | 128               | 512            |
| Model size     | ~37 MB            | ~66 MB (int8)  |
| Default cutoff | 0.363 (published) | 0.239 (derived)  |
| LFW accuracy   | 0.9887            | **0.9932**       |

Both backends detect with YuNet and compare by cosine similarity; they
differ only in how a detected face becomes numbers. Measured over LFW's
full 6000-pair protocol through this tool's own pipeline, ArcFace
scores 0.9932 balanced accuracy against SFace's 0.9887, and separates
the classes further — mean similarity 0.6768 vs 0.0045 for ArcFace,
0.6491 vs 0.0833 for SFace. Most of the advantage is in scoring
non-matches lower.

**The two cutoffs are not interchangeable.** A threshold is a property
of the model that produced the embeddings. Passing SFace's 0.363 to
ArcFace would miss 56 of 3000 genuine LFW pairs where 0.239 misses 37
— it wouldn't fail loudly, it would just quietly reject matches, the
same way `--threshold 1.0` once reported an identical face as "0.0%
Almost no chance".

#### How ArcFace's 0.239 was derived

Over LFW's 6000 pairs — 3000 same, 3000 different, across its ten
official folds — with every pair usable and none skipped for a missed
detection. The result was **0.2394 ± 0.0087** at **0.9932 ± 0.0034**
balanced accuracy.

Two conditions make that trustworthy, and the second is the one that
is easy to miss:

1. The per-fold spread is small.
2. **No threshold anywhere scores a perfect 1.0000**, which means the
   set contains pairs hard enough to actually pin the value.

Without the second check, a small easy set produces a confident and
worthless answer. A 66-image set tried here admitted *every* threshold
from 0.167 to 0.729 at a perfect score — a plateau 0.563 wide — and
duly reported the midpoint of that empty gap as 0.448 ± 0.0043, with
no warning. On LFW the plateau is 0.199 wide and nothing scores
perfectly.

That is also why 0.239 is so much lower than every earlier estimate
(0.374, 0.40, 0.448, 0.493). Small sets contain no hard genuine pairs,
so nothing pulls the cut downward and it drifts up into the gap. At
0.40 this backend missed 73 of 3000 genuine pairs; at 0.239 it misses
37.

SFace's published 0.363 was checked the same way. Deriving it through
this pipeline puts the optimum slightly lower, at 0.3156 (55 missed
pairs against 0.363's 72), but the difference is 0.13 percentage
points of balanced accuracy, so the published value is kept rather
than changing a shipped default for that.

`tools/derive_threshold.py` is what derives it. It runs the same
pipeline `facecomp` does — YuNet detect, 5-point align, embed, cosine
— over a labelled pair set, then picks the threshold by 10-fold
cross-validation, choosing it on nine folds and scoring it on the
tenth. That reports how stable the value is rather than just what it
is, so one lucky split can't pass for a result:

```sh
# LFW, standard protocol (pairs.txt + lfw/<Name>/<Name>_0001.jpg)
python3 tools/derive_threshold.py \
  --images-dir /path/to/lfw --pairs /path/to/pairs.txt \
  --detector face_detection_yunet_2023mar.onnx \
  --recognizer arcfaceresnet100-11-int8.onnx \
  --backend arcface --det-score 0.5

# any second dataset, as <imgA>\t<imgB>\t<1|0>
python3 tools/derive_threshold.py --pairs-format tsv ...
```

Needs `opencv-python` (4.10 or newer) and `numpy`. Two details that
are easy to get wrong and are baked in: it scores by *balanced*
accuracy, because plain accuracy rates an "always different"
classifier at 87% on a typical imbalanced pair set; and it assigns
folds stratified, because with only a few dozen positive pairs
sequential folds can contain no same-person pair at all, making that
fold's score meaningless. It prints a warning when the per-fold spread
exceeds 0.05 — heed it. That is exactly what disqualified the two
small sets above.

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
  /path/to/face_recognition_sface_2021dec.onnx \
  /path/to/arcfaceresnet100-11-int8.onnx
```

The third argument is optional. Passing it bundles ArcFace too, so the
AppImage supports `--backend arcface`; this adds ~66 MB, taking the
result to roughly 105 MB. Omit it for a smaller SFace-only bundle, in
which case `--backend arcface` reports that no model is available
rather than failing deeper in OpenCV.

This builds OpenCV (core/imgproc/imgcodecs/objdetect/dnn/calib3d only,
to keep build time down — expect roughly 10 minutes on a few cores,
cached across runs in CI) and `facecomp` in release mode, fetches
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
  `FACECOMP_DETECTOR_MODEL`/`FACECOMP_ENCODER_MODEL`/`FACECOMP_ARCFACE_MODEL`
  if you want to point at different ones). Both recognition models are
  exported up front and `--backend` picks between them at run time,
  since `AppRun` can't know which one you'll ask for.
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
compared against each other), and results are reported closest match
first:

```
master: master.jpg
backend: sface
embedding: 128 dimensions per face

photo                           faces similarity  match %  confidence
photo1.jpg                          1     0.7437    79.9%  Likely
photo2.jpg                          1     0.0992    29.3%  Unlikely
```

The `embedding` line reports how many numbers each face was reduced to
before comparison — see [How faces are compared](#how-faces-are-compared).
It's read back from the model actually in use rather than hardcoded, so
it stays accurate if you pass a different recognition model.

### Several master photos

`--master` also takes more than one photo of the same person. Their
embeddings are averaged into a single template, which matches more
reliably than any one photo:

```sh
facecomp --master alice-1.jpg alice-2.jpg alice-3.jpg --slave photos/*.jpg
```

```
master: 3 photos averaged into one template
  alice-1.jpg
  alice-2.jpg
  alice-3.jpg
agreement: 0.7838 (lowest similarity between master photos)
```

A single photograph is one noisy sample of a face — it carries that
day's lighting, that angle, that expression as well as the person.
Averaging several cancels the nuisance variation and leaves what they
share. Measured leave-one-out over three photographs of one subject
(enrol two, probe the held-out third, against seven other people), the
template scored the genuine probe higher than **either** enrolment photo
in all three folds:

| held-out probe | enrolment photo A | enrolment photo B | template |
|---|---|---|---|
| 1 | 0.7747 | 0.8035 | **0.8402** |
| 2 | 0.7747 | 0.7639 | **0.8101** |
| 3 | 0.8035 | 0.7639 | **0.8320** |

Mean margin to the nearest impostor went 0.6556 → 0.6934.

The gain comes from *genuine* variety, so pick photos that differ —
different sessions, angles, expressions. Near-duplicates embed almost
identically, so their average barely moves from any one of them and buys
nothing.

**The `agreement` line is a safeguard, and worth reading.** It reports
the lowest similarity between any two master photos. Enrolling several
photos asserts they are all the same person, and nothing else checks
that: a stray photo of someone else drags the template toward them and
then quietly mis-scores every comparison made against it. A disagreeing
pair is the only visible symptom, so it's printed every time, with a
warning when it falls below `--threshold`.

Averaging was also compared against keeping every photo and taking each
one's best match ("max over set"). Averaging won on every fold, for two
reasons that aren't artifacts of the small sample: a maximum can only
ever pick the best enrolment photo, whereas an average can beat all of
them by cancelling noise; and a maximum raises impostor scores by the
same mechanism, since an impostor need only resemble one photo. That
option was measured, rejected, and removed rather than shipped as a
tuning knob.

One caveat. A template raises genuine scores by design, so the default
cutoffs — derived for one-to-one matching — are more permissive here
than they were calibrated to be. LFW's pair protocol is one-photo-to-
one-photo, so it cannot derive a template threshold; that needs a set
with several photos per identity on both sides. Until then, treat
multi-photo results as *better ordered* rather than better calibrated,
and lean on the similarity column rather than the match percentage.

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
`FACECOMP_DETECTOR_MODEL`, and `FACECOMP_ENCODER_MODEL` or
`FACECOMP_ARCFACE_MODEL` depending on `--backend`.

Other flags:

- `--master <photo>...` — takes one photo or several of the same
  person. Several are averaged into one template; see
  [Several master photos](#several-master-photos).
- `--backend <sface|arcface>` — which model turns a detected face into
  numbers (default `sface`). `--encoder-model` must be the matching
  weights. See [Choosing a backend](#choosing-a-backend).
- `--threshold <f64>` — the cosine similarity at/above which two faces
  count as the same person (default `0.363` for `sface`, the OpenCV
  Zoo-published recommendation; `0.239` for `arcface`, derived over
  LFW — see [Choosing a backend](#choosing-a-backend)). Must be greater
  than 0 and less than 1;
  cosine similarity never exceeds 1.0, and a threshold at or above it
  would collapse the match-percent scale rather than simply matching
  nothing, so such values are rejected outright.
- `--max <n>` — report only the `n` closest-matching photos instead of
  every one compared, useful when `--slave` expands to a large
  directory. Since results are always ordered best match first, this
  just trims the tail. Must be at least 1, since `--max 0` would report
  nothing at all rather than erroring.

  It affects presentation only: every photo is still detected, embedded
  and compared, so warnings and the exit status still account for
  photos that fall outside the `n` shown.
- `--detection-confidence <f32>` — how confident YuNet must be before a
  candidate counts as a face (default `0.7`; also settable via
  `FACECOMP_DETECTION_CONFIDENCE`). Lower it if `facecomp` reports "no
  face detected" on photos that clearly contain one; raise it if it
  picks up things that aren't faces. Must be greater than 0 and at most
  1 — it's a probability, and 0 accepts every candidate the network
  proposes (a single portrait yielded 1543 "faces"). See
  [Detection confidence](#detection-confidence).
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

### How faces are compared

Each detected face is reduced to an **embedding** — a fixed-length list
of numbers describing that face. Two faces are compared by the cosine
similarity between their embeddings, which is the `similarity` column.
How many numbers depends on `--backend` — **128 per face** for SFace,
**512** for ArcFace — reported as the `embedding` line in the output
(`embedding_dimensions` in `--json`).

Those values are what's actually compared. It's worth separating them
from a different count that often gets conflated with them: the
detector also locates **5 facial landmarks** (both eyes, the nose tip,
and both mouth corners), but those exist only to align and crop a face
into a canonical position before it's embedded. They are never compared
against each other, so a detector reporting more landmarks would not
mean more points of comparison — only more precise alignment.

If you want more points of comparison, that means a recognition model
with a larger embedding — which is what `--backend arcface` selects —
not a detector with more landmarks.

Alignment is where those 5 landmarks earn their keep, and it is more
delicate than it looks. SFace aligns internally via OpenCV's
`alignCrop`. ArcFace is a bare ONNX graph, so `facecomp` aligns for it:
the landmarks are fitted onto ArcFace's canonical 112×112 layout with
`estimateAffinePartial2D` and the face warped onto it. YuNet emits its
landmarks in the same order that layout is written in, so they map
across directly with no reordering — and it has to stay that way.
Getting the order wrong doesn't error, doesn't look broken, and doesn't
even degrade uniformly: feeding them in mirrored order (both eyes and
both mouth corners swapped, which is what assuming the opposite
handedness convention gives you) scored a *different* person at 0.573
against the same person's 0.460 — an inverted verdict, reported with
full confidence. `tests/arcface_alignment.rs` pins the ordering, and
`examples/ordering_check.rs` is the tool that measured it end to end.

### Detection confidence

Before anything can be compared, YuNet has to actually find a face.
`--detection-confidence` sets how sure it must be, and the default of
`0.7` is deliberately lower than OpenCV's own `0.9`, which is too
strict for ordinary photographs. Measured over a 64-image set of
real-world photos:

| `--detection-confidence` | Photos where a face was found | False detections on face-free images |
|---|---|---|
| 0.9 (OpenCV's default) | 41/64 (64%) | none |
| 0.8                    | 48/64 (75%) | none |
| **0.7 (facecomp's default)** | **59/64 (92%)** | one (a photo of dogs) |
| 0.6                    | 63/64 (98%) | two |
| 0.5                    | 64/64 (100%) | two |

0.7 is where that curve turns: below it, the detector starts firing on
non-face imagery and same/different separation begins to degrade, while
above it a lot of perfectly ordinary photos are simply refused.

**Which value to use.** Run `0.8` when every result needs to be
trustworthy — it never picked up a non-face, and still finds faces in
more photos than `0.9` does, so there is no reason to run `0.9` at all.
Keep the `0.7` default when you would rather not silently skip photos.
Don't go below `0.6`.

Note this setting doesn't govern how *accurate* a comparison is — that
comes from the model and `--threshold`. What it changes is which photos
produce a face at all. It does have a second-order effect worth knowing:
a marginal detection gives sloppier landmarks, so the face is aligned
less precisely before embedding, which makes that particular result less
reliable. That's a good reason to re-check a borderline result (an "Even
chance" verdict, or a surprising `faces` count) at `0.8`.

The two kinds of mistake are not equally costly, which is why the
default leans toward sensitivity. A missed face is a hard failure —
the photo can't be compared at all and you get `no face detected`. A
spurious detection just adds a row with a low similarity score: a dog
photo compared against a person lands around 14% "Very unlikely",
which reads correctly rather than misleadingly.

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
  ;; Only needed if `facecomp-executable` doesn't already know where its
  ;; own model files are - the AppImage build does, via bundled
  ;; FACECOMP_DETECTOR_MODEL/FACECOMP_ENCODER_MODEL defaults, so if
  ;; you're running that, leave these two unset:
  (facecomp-detector-model "/path/to/face_detection_yunet_2023mar.onnx")
  (facecomp-encoder-model "/path/to/face_recognition_sface_2021dec.onnx")
  ;; Only needed if the `facecomp` binary isn't already on PATH:
  (facecomp-executable "/path/to/facecomp/target/release/facecomp")
  ;; Optional: leave unset to use facecomp's own default (0.7). Set it
  ;; only to override - see "Detection confidence" above.
  (facecomp-detection-confidence 0.7))
```

Every setting above is optional and defaults to "let `facecomp`
decide", so nothing here pins a value that the binary might later
change out from under it.

To use ArcFace instead, set the backend and point at its weights —
that's all:

```elisp
(facecomp-backend 'arcface)
(facecomp-arcface-model "/path/to/arcfaceresnet100-11-int8.onnx")
```

Leave `facecomp-threshold` nil, which is the recommended setting. The
executable then applies the cutoff belonging to whichever backend is
selected — 0.363 for SFace, 0.239 for ArcFace — so the two can't get
crossed. Pinning ArcFace's backend while leaving SFace's threshold
pinned wouldn't error; it would quietly reject genuine matches.

`facecomp-encoder-model` and `facecomp-arcface-model` are separate
settings on purpose: whichever one matches `facecomp-backend` is the
one passed, so the weights can't end up disagreeing with the backend
they're loaded for. `facecomp-max` mirrors `--max` if you want only
the closest few results.

Then:

- `M-x facecomp-compare` prompts for a master photo, then for the
  photos to compare against it, picked one at a time. Picking starts
  in the master photo's own directory rather than wherever you invoked
  `M-x` from. To select many photos at once, use the Dired route below.
- Or mark two or more files in Dired and run `M-x facecomp-compare` —
  you'll be prompted for which of the marked files is the master
  (defaulting to the topmost one in the Dired listing), and the rest
  are compared against it. This is a real prompt rather than "whichever
  one you marked first," since Dired itself doesn't track marking
  order - only buffer order.
- `C-u M-x facecomp-compare` additionally prompts for a detection
  confidence to use for that run only (defaulting to `0.8`), leaving
  `facecomp-detection-confidence` untouched. This is the Emacs
  equivalent of passing `--detection-confidence` on the command line,
  and it's the convenient way to re-check a borderline result at a
  stricter setting — see [Detection confidence](#detection-confidence)
  — without editing your configuration and changing it back.

Results are shown in a `*facecomp*` buffer, one photo per entry, with
the match percentage and confidence label colored by confidence.

## License

GPL-3.0-or-later. See `LICENSE`.
