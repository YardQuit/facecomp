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
;; each photo, computes a 128-dimension embedding with dlib, and reports
;; a distance-based percentage match between every pair of photos.
;;
;; `facecomp' itself works standalone from a shell; this file just
;; shells out to it and renders the result in an Emacs buffer.
;;
;; Usage:
;; - `M-x facecomp-compare' prompts for two or more image files.
;; - Called from Dired with two or more files marked, it compares the
;;   marked files instead of prompting.
;;
;; Setup:
;; Set `facecomp-landmark-model' and `facecomp-encoder-model' to the
;; paths of dlib's model files before first use (see the README for
;; where to get them), and make sure the `facecomp' executable is on
;; PATH or set `facecomp-executable' to its full path.

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

(defcustom facecomp-landmark-model nil
  "Path to dlib's `shape_predictor_68_face_landmarks.dat'."
  :type '(choice (const :tag "Not set" nil) file)
  :group 'facecomp)

(defcustom facecomp-encoder-model nil
  "Path to dlib's `dlib_face_recognition_resnet_model_v1.dat'."
  :type '(choice (const :tag "Not set" nil) file)
  :group 'facecomp)

(defcustom facecomp-threshold 0.6
  "Euclidean distance at/below which two faces count as the same person.
Passed through to the `facecomp' executable's `--threshold' flag."
  :type 'float
  :group 'facecomp)

(defun facecomp--require-models ()
  "Signal a `user-error' unless both model paths are configured and exist."
  (unless (and facecomp-landmark-model (file-exists-p facecomp-landmark-model))
    (user-error "Set `facecomp-landmark-model' to dlib's shape predictor file"))
  (unless (and facecomp-encoder-model (file-exists-p facecomp-encoder-model))
    (user-error "Set `facecomp-encoder-model' to dlib's face recognition model file")))

(defun facecomp--read-image-files ()
  "Prompt for two or more existing image files and return them as a list."
  (let ((files (list (read-file-name "Image file 1: " nil nil t)
                      (read-file-name "Image file 2: " nil nil t))))
    (while (y-or-n-p (format "Add another image (%d selected so far)? " (length files)))
      (push (read-file-name (format "Image file %d: " (1+ (length files))) nil nil t)
            files))
    (mapcar #'expand-file-name (nreverse files))))

(defun facecomp--run (files)
  "Run the facecomp executable on FILES and return the parsed JSON report."
  (facecomp--require-models)
  (unless (executable-find facecomp-executable)
    (user-error "Could not find `%s' on PATH; set `facecomp-executable'"
                facecomp-executable))
  (with-temp-buffer
    (let* ((args (append (list "--landmark-model" facecomp-landmark-model
                                "--encoder-model" facecomp-encoder-model
                                "--threshold" (number-to-string facecomp-threshold)
                                "--json")
                         files))
           (status (apply #'call-process facecomp-executable nil t nil args))
           (output (buffer-string)))
      (condition-case _
          (json-parse-string output :object-type 'alist :array-type 'list)
        (error (error "facecomp exited with status %s: %s" status output))))))

(defun facecomp--percent-face (percent)
  "Return a face symbol reflecting how confident a PERCENT match is."
  (cond ((>= percent 80) 'success)
        ((>= percent 50) 'warning)
        (t 'error)))

(defun facecomp--render (report)
  "Render a parsed facecomp REPORT into the *facecomp* buffer."
  (let ((buf (get-buffer-create "*facecomp*")))
    (with-current-buffer buf
      (let ((inhibit-read-only t))
        (erase-buffer)
        (dolist (err (alist-get 'errors report))
          (insert (propertize (format "warning: %s\n" err) 'face 'warning)))
        (when (alist-get 'errors report)
          (insert "\n"))
        (let ((pairs (alist-get 'pairs report)))
          (if (null pairs)
              (insert "No comparable pairs (see warnings above).\n")
            (dolist (pair pairs)
              (let ((a (alist-get 'image_a pair))
                    (b (alist-get 'image_b pair))
                    (pct (alist-get 'match_percent pair))
                    (same (alist-get 'same_person pair)))
                (insert (format "%s\n  vs %s\n  " a b))
                (insert (propertize (format "%.1f%% match" pct)
                                     'face (facecomp--percent-face pct)))
                (insert (format " (%s)\n\n"
                                (if (eq same t) "same person" "different people")))))))
        (goto-char (point-min))
        (special-mode)))
    (display-buffer buf)))

;;;###autoload
(defun facecomp-compare (files)
  "Compare two or more image FILES and show a percentage match per pair.
When called from a Dired buffer with two or more files marked, compares
those files.  Otherwise prompts for image files interactively."
  (interactive
   (list (if (and (derived-mode-p 'dired-mode)
                  (>= (length (dired-get-marked-files)) 2))
             (dired-get-marked-files)
           (facecomp--read-image-files))))
  (when (< (length files) 2)
    (user-error "Select at least two images"))
  (facecomp--render (facecomp--run files)))

(provide 'facecomp)

;;; facecomp.el ends here
