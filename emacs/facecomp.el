;;; facecomp.el --- Compare faces across photos via a Rust backend -*- lexical-binding: t -*-

;; Copyright (C) 2026 Michael Jones
;; Author: Michael Jones <yardquit@pm.me>
;; Maintainer: Michael Jones
;; Assisted-by: Claude [Claude Code]
;; URL: https://github.com/yardquit/facecomp
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: multimedia, convenience
;; Homepage: https://github.com/yardquit/facecomp

;; This file is not part of GNU Emacs.
;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program. If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:
;; Emacs frontend for the `facecomp' command-line tool (the Rust crate
;; in this repository).  `facecomp' detects the most prominent face in
;; a master photo and in each of one or more other photos, computes a
;; 128-dimension embedding for each with OpenCV's SFace, and reports a
;; similarity-based percentage match plus a qualitative confidence label
;; between the master and every other photo.
;;
;; `facecomp' itself works standalone from a shell; this file just
;; shells out to it and renders the result in an Emacs buffer.
;;
;; Usage:
;; - `M-x facecomp-compare' prompts for a master photo, then for the
;;   photos to compare against it, picked one at a time.
;; - Called from Dired with two or more files marked, the first marked
;;   file is used as the master and the rest as the photos to compare
;;   against it.  This is the way to select many photos at once.
;;
;; Setup:
;; Make sure the `facecomp' executable is on PATH, or set
;; `facecomp-executable' to its full path. If your `facecomp' doesn't
;; already know where its model files are - e.g. it's a plain build
;; rather than the AppImage, which bakes its own model paths in - also
;; set `facecomp-detector-model' and `facecomp-encoder-model' to the
;; paths of OpenCV Zoo's model files (see the README for where to get
;; them).

;;; Code:

(require 'json)
(require 'dired)

(defgroup facecomp nil
  "Compare faces across photos using the facecomp Rust backend."
  :group 'multimedia
  :prefix "facecomp-")

(defcustom facecomp-executable "facecomp"
  "Path to the `facecomp' executable."
  :type 'string
  :group 'facecomp)

(defcustom facecomp-detector-model nil
  "Path to OpenCV Zoo's `face_detection_yunet_2023mar.onnx'.
Leave this nil if `facecomp-executable' already knows where to find
its own model file - the AppImage build does, via its bundled
`FACECOMP_DETECTOR_MODEL' default - in which case the `--detector-model'
flag is simply omitted and the executable's own default is used."
  :type '(choice (const :tag "Let facecomp decide" nil) file)
  :group 'facecomp)

(defcustom facecomp-encoder-model nil
  "Path to OpenCV Zoo's `face_recognition_sface_2021dec.onnx'.
Leave this nil if `facecomp-executable' already knows where to find
its own model file - the AppImage build does, via its bundled
`FACECOMP_ENCODER_MODEL' default - in which case the `--encoder-model'
flag is simply omitted and the executable's own default is used."
  :type '(choice (const :tag "Let facecomp decide" nil) file)
  :group 'facecomp)

(defcustom facecomp-threshold 0.363
  "Cosine similarity at/above which two faces count as the same person.
Passed through to the `facecomp' executable's `--threshold' flag."
  :type 'float
  :group 'facecomp)

(defcustom facecomp-detection-confidence nil
  "Detector confidence at/above which a candidate counts as a face.
Lower it to find faces in difficult photos, raise it if non-faces are
being picked up.  Leave nil to use the `facecomp' executable's own
default, so the two can't drift apart when that default changes."
  :type '(choice (const :tag "Let facecomp decide" nil) float)
  :group 'facecomp)

(defun facecomp--model-args ()
  "Build the `--detector-model'/`--encoder-model' flags, if configured.
Each is included only when its customize variable is set to an
existing file; otherwise it's left out entirely so the executable
falls back to its own default (e.g. the AppImage build's bundled
models). A variable that IS set but points nowhere is treated as a
user mistake worth catching early, rather than silently ignored."
  (let (args)
    (when facecomp-detector-model
      (unless (file-exists-p facecomp-detector-model)
        (user-error "`facecomp-detector-model' is set to `%s', which doesn't exist"
                    facecomp-detector-model))
      (setq args (nconc args (list "--detector-model" facecomp-detector-model))))
    (when facecomp-encoder-model
      (unless (file-exists-p facecomp-encoder-model)
        (user-error "`facecomp-encoder-model' is set to `%s', which doesn't exist"
                    facecomp-encoder-model))
      (setq args (nconc args (list "--encoder-model" facecomp-encoder-model))))
    args))

(defun facecomp--read-targets (&optional master)
  "Prompt for photos to compare against the master, one at a time.

Picking starts in MASTER's own directory, which is where the photos
being compared almost always live - rather than in `default-directory',
which is just wherever \\[execute-extended-command] happened to be
invoked from.

For selecting many photos at once, mark them in Dired and call
`facecomp-compare' from there."
  (let ((dir (or (and master (file-name-directory master)) default-directory))
        files)
    (push (read-file-name "Photo: " dir nil t) files)
    (while (y-or-n-p (format "Add another photo (%d selected so far)? " (length files)))
      (push (read-file-name (format "Photo %d: " (1+ (length files))) dir nil t) files))
    (mapcar #'expand-file-name (nreverse files))))

(defun facecomp--run (master targets &optional confidence)
  "Run the facecomp executable comparing MASTER against TARGETS.
CONFIDENCE, when non-nil, overrides `facecomp-detection-confidence' for
this run only; the customize variable itself is left untouched.
Returns the parsed JSON report."
  (unless (executable-find facecomp-executable)
    (user-error "Could not find `%s' on PATH; set `facecomp-executable'"
                facecomp-executable))
  (with-temp-buffer
    (let* ((conf (or confidence facecomp-detection-confidence))
           (args (append (list "--master" master)
                          (facecomp--model-args)
                          (list "--threshold" (number-to-string facecomp-threshold))
                          (when conf
                            (list "--detection-confidence" (number-to-string conf)))
                          (list "--json" "--slave")
                          targets))
           ;; Keep stderr out of the buffer we parse: a bare `t' destination
           ;; merges both streams into it. Held aside so it can still be
           ;; shown if the run fails.
           (stderr-file (make-temp-file "facecomp-stderr"))
           status output stderr)
      (unwind-protect
          (progn
            (setq status (apply #'call-process facecomp-executable nil
                                (list t stderr-file) nil args))
            (setq output (buffer-string))
            (setq stderr (with-temp-buffer
                           (insert-file-contents stderr-file)
                           (string-trim (buffer-string)))))
        (delete-file stderr-file))
      (condition-case _
          (facecomp--parse-report output)
        (error
         (error "facecomp exited with status %s: %s%s" status output
                (if (string-empty-p stderr) "" (concat "\nstderr: " stderr))))))))

(defun facecomp--parse-report (output)
  "Parse OUTPUT as facecomp's JSON report, tolerating leading noise.

Parsing starts at the first brace rather than at the first character,
because the AppImage runtime can print a diagnostic to *stdout* - not
stderr - ahead of the payload it launches. On a host without FUSE it
says \"No suitable fusermount binary found on the $PATH\", then runs
anyway by self-extracting; the run succeeds and the report is correct,
but that line sits in front of it. Anchoring to the brace keeps such
noise from failing an otherwise good run."
  (let ((start (string-match "{" output)))
    (unless start
      (error "no JSON object in facecomp output"))
    (json-parse-string (substring output start)
                       :object-type 'alist :array-type 'list)))

(defun facecomp--percent-face (percent)
  "Return a face symbol reflecting how confident a PERCENT match is."
  (cond ((>= percent 80) 'success)
        ((>= percent 45) 'warning)
        (t 'error)))

(defun facecomp--render (report)
  "Render a parsed facecomp REPORT into the *facecomp* buffer."
  (let ((buf (get-buffer-create "*facecomp*")))
    (with-current-buffer buf
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (format "master: %s\n" (alist-get 'master report)))
        ;; Older facecomp builds don't report this, so only show it when present.
        (let ((dims (alist-get 'embedding_dimensions report)))
          (when dims
            (insert (format "embedding: %d dimensions per face\n" dims))))
        (insert "\n")
        (dolist (err (alist-get 'errors report))
          (insert (propertize (format "warning: %s\n" err) 'face 'warning)))
        (when (alist-get 'errors report)
          (insert "\n"))
        (let ((results (alist-get 'results report)))
          (if (null results)
              (insert "No comparable photos (see warnings above).\n")
            (dolist (r results)
              (let ((photo (alist-get 'photo r))
                    (faces (alist-get 'faces_detected r))
                    (pct (alist-get 'match_percent r))
                    (confidence (alist-get 'confidence r)))
                (insert (format "%s\n  " photo))
                (insert (propertize (format "%.1f%% match — %s" pct confidence)
                                     'face (facecomp--percent-face pct)))
                (when (and faces (> faces 1))
                  (insert (format "  [best of %d faces in photo]" faces)))
                (insert "\n\n")))))
        (goto-char (point-min))
        (special-mode)))
    (display-buffer buf)))

(defun facecomp--choose-master (marked)
  "Prompt for which file in MARKED is the master; return (MASTER TARGETS).
MARKED's own ordering is Dired's buffer order, not the order the
files were actually marked in - Dired doesn't track that at all - so
this always asks explicitly rather than guessing from position."
  (let* ((candidates (mapcar (lambda (f) (cons (file-name-nondirectory f) f)) marked))
         (choice (completing-read
                  (format "Master photo (of %d marked): " (length marked))
                  candidates nil t nil nil (caar candidates)))
         (master (alist-get choice candidates nil nil #'string=)))
    (list master (remove master marked))))

(defun facecomp--read-confidence ()
  "Prompt for a one-off detection confidence, and sanity-check it.
Defaults to 0.8, the value facecomp documents for when every result
needs to be trustworthy - which is the usual reason to re-run a
borderline comparison at a different setting."
  (let ((conf (read-number "Detection confidence for this run: " 0.8)))
    (unless (and (> conf 0.0) (<= conf 1.0))
      (user-error "Detection confidence must be greater than 0 and at most 1"))
    conf))

;;;###autoload
(defun facecomp-compare (master targets &optional confidence)
  "Compare MASTER against each of TARGETS and show a percentage match.
Each result also gets a qualitative confidence label (Almost certain,
Very likely, Likely, ...).

When called from a Dired buffer with two or more files marked, prompts
for which of the marked files is MASTER (defaulting to the topmost one
in the buffer) and uses the rest as TARGETS. That is the way to select
many photos at once. Otherwise prompts for a master photo, then for
the photos to compare against it, picked one at a time starting in the
master's own directory.

With a prefix argument, also prompts for a detector CONFIDENCE to use
for this run only, leaving `facecomp-detection-confidence' alone. Use
it to re-check a borderline result at a stricter setting without
having to change your configuration and change it back."
  (interactive
   (append
    (if (and (derived-mode-p 'dired-mode)
             (>= (length (dired-get-marked-files)) 2))
        (facecomp--choose-master (dired-get-marked-files))
      (let ((master (expand-file-name (read-file-name "Master photo: " nil nil t))))
        (list master (facecomp--read-targets master))))
    ;; Read last, so the prefix-arg prompt doesn't precede choosing photos.
    (list (when current-prefix-arg (facecomp--read-confidence)))))
  (when (null targets)
    (user-error "Select at least one photo to compare against the master"))
  (facecomp--render (facecomp--run master targets confidence)))

(provide 'facecomp)

;;; facecomp.el ends here
