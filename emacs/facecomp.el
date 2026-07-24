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
;; - `M-x facecomp-compare' prompts for a master photo, then either a
;;   glob pattern (e.g. "*.png") or image files picked one at a time.
;; - Called from Dired with two or more files marked, the first marked
;;   file is used as the master and the rest as the photos to compare
;;   against it.
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

(defun facecomp--read-targets ()
  "Prompt for a glob pattern, or, if left blank, files picked one at a time."
  (let ((pattern (read-string
                  "Photos to compare against master (glob, e.g. *.png; blank to pick one at a time): ")))
    (if (not (string-empty-p pattern))
        (or (file-expand-wildcards pattern t)
            (user-error "No files matched `%s'" pattern))
      (let (files)
        (push (read-file-name "Photo: " nil nil t) files)
        (while (y-or-n-p (format "Add another photo (%d selected so far)? " (length files)))
          (push (read-file-name (format "Photo %d: " (1+ (length files))) nil nil t) files))
        (mapcar #'expand-file-name (nreverse files))))))

(defun facecomp--run (master targets)
  "Run the facecomp executable comparing MASTER against TARGETS.
Returns the parsed JSON report."
  (unless (executable-find facecomp-executable)
    (user-error "Could not find `%s' on PATH; set `facecomp-executable'"
                facecomp-executable))
  (with-temp-buffer
    (let* ((args (append (list "--master" master)
                          (facecomp--model-args)
                          (list "--threshold" (number-to-string facecomp-threshold)
                                "--json"
                                "--slave")
                          targets))
           (status (apply #'call-process facecomp-executable nil t nil args))
           (output (buffer-string)))
      (condition-case _
          (json-parse-string output :object-type 'alist :array-type 'list)
        (error (error "facecomp exited with status %s: %s" status output))))))

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
        (insert (format "master: %s\n\n" (alist-get 'master report)))
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
                    (confidence (alist-get 'confidence r))
                    (same (alist-get 'same_person r)))
                (insert (format "%s\n  " photo))
                (insert (propertize (format "%.1f%% match — %s" pct confidence)
                                     'face (facecomp--percent-face pct)))
                (insert (format " (%s)"
                                (if (eq same t) "same person" "different people")))
                (when (and faces (> faces 1))
                  (insert (format "  [best of %d faces in photo]" faces)))
                (insert "\n\n")))))
        (goto-char (point-min))
        (special-mode)))
    (display-buffer buf)))

;;;###autoload
(defun facecomp-compare (master targets)
  "Compare MASTER against each of TARGETS and show a percentage match.
Each result also gets a qualitative confidence label (Almost certain,
Very likely, Likely, ...).

When called from a Dired buffer with two or more files marked, the
first marked file is used as MASTER and the rest as TARGETS.
Otherwise prompts for a master photo, then either a glob pattern
\(e.g. \"*.png\"\) or photos picked one at a time."
  (interactive
   (if (and (derived-mode-p 'dired-mode)
            (>= (length (dired-get-marked-files)) 2))
       (let ((marked (dired-get-marked-files)))
         (list (car marked) (cdr marked)))
     (let ((master (expand-file-name (read-file-name "Master photo: " nil nil t))))
       (list master (facecomp--read-targets)))))
  (when (null targets)
    (user-error "Select at least one photo to compare against the master"))
  (facecomp--render (facecomp--run master targets)))

(provide 'facecomp)

;;; facecomp.el ends here
