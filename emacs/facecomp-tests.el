;;; facecomp-tests.el --- Tests for facecomp.el -*- lexical-binding: t -*-

;; Copyright (C) 2026 Michael Jones
;; Author: Michael Jones <yardquit@pm.me>
;; Assisted-by: Claude [Claude Code]

;; This file is not part of GNU Emacs.
;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;;; Commentary:
;; ERT suite for the Emacs front end.  Run it from the repository root:
;;
;;   emacs -Q --batch -L emacs -l emacs/facecomp-tests.el \
;;         -f ert-run-tests-batch-and-exit
;;
;; Nothing here runs the `facecomp' executable, downloads a model or
;; touches a photograph.  `call-process' is stubbed and the argument
;; vector it would have been handed is inspected instead, which is the
;; part of this file that can actually be wrong: everything the Rust
;; side gets told about a comparison - which cutoff, which backend,
;; which weights, which photos are masters - is decided here, in Lisp,
;; and a mistake in it produces a confident wrong number rather than an
;; error.  That is the same reason `tests/arcface_alignment.rs' pins
;; landmark ordering four ways.
;;
;; Every option is rebound to its documented default around each test
;; by `facecomp-test--with-defaults', so a test cannot pass because of
;; the customize settings of whoever is running it, and cannot leak a
;; setting into the next test.

;;; Code:

(require 'ert)
(require 'cl-lib)
(require 'facecomp)

(defconst facecomp-test--directory
  (file-name-directory (or load-file-name buffer-file-name default-directory))
  "Directory this file was loaded from.

Captured now rather than inside a test: `load-file-name' is bound only
while a file is being loaded, and is nil by the time ERT runs anything.")

;;;; Fixtures

(defmacro facecomp-test--with-defaults (&rest body)
  "Run BODY with every facecomp option bound to its documented default.

These are `defcustom' variables, so they are global state: without
this a test would be reading whatever the person running the suite
happens to have set, and would leave its own settings behind for the
next test to trip over."
  (declare (indent 0) (debug t))
  `(let ((facecomp-executable "facecomp")
         (facecomp-detector-model nil)
         (facecomp-encoder-model nil)
         (facecomp-backend nil)
         (facecomp-arcface-model nil)
         (facecomp-threshold nil)
         (facecomp-max nil)
         (facecomp-detection-confidence nil))
     ,@body))

(defmacro facecomp-test--with-temp-files (vars &rest body)
  "Bind each symbol in VARS to a fresh existing temp file around BODY.

`facecomp--model-args' checks that a configured model path exists, so
tests about which flags get built need paths that really are there -
otherwise they would pass by taking the not-found branch."
  (declare (indent 1) (debug t))
  (if (null vars)
      `(progn ,@body)
    `(let ((,(car vars) (make-temp-file "facecomp-test-model")))
       (unwind-protect
           (facecomp-test--with-temp-files ,(cdr vars) ,@body)
         (delete-file ,(car vars))))))

(defvar facecomp-test--process-args nil
  "Arguments the stubbed `call-process' was last called with.")

(defmacro facecomp-test--with-stub-process (stdout status &rest body)
  "Run BODY with `call-process' stubbed to emit STDOUT and return STATUS.

The stub records its arguments in `facecomp-test--process-args'.
`executable-find' is stubbed alongside it because `facecomp--run'
refuses to go any further without it, and the point of these tests is
what happens after that check rather than the check itself."
  (declare (indent 2) (debug t))
  `(let ((facecomp-test--process-args nil))
     (cl-letf (((symbol-function 'executable-find)
                (lambda (&rest _) "/nonexistent/facecomp"))
               ((symbol-function 'call-process)
                (lambda (_program _infile _destination _display &rest args)
                  (setq facecomp-test--process-args args)
                  (insert ,stdout)
                  ,status)))
       ,@body)))

(defun facecomp-test--render (report)
  "Render REPORT and return the *facecomp* buffer's contents, properties and all."
  (when (get-buffer "*facecomp*")
    (kill-buffer "*facecomp*"))
  (facecomp--render report)
  (with-current-buffer "*facecomp*" (buffer-string)))

(defun facecomp-test--face-at (string needle)
  "Return the `face' property where NEEDLE starts in STRING."
  (let ((at (string-match (regexp-quote needle) string)))
    (should at)
    (get-text-property at 'face string)))

;;;; The options themselves

(ert-deftest facecomp-test-every-optional-setting-defaults-to-nil ()
  "Nil means \"don't pass the flag\", which is what keeps Lisp and Rust in step.

A numeric default here would be a second copy of a number that lives
in `src/lib.rs', free to drift from it.  `facecomp-threshold' is the
one that matters most: a cutoff is a property of the recognition
model, so hardcoding one would pin SFace's 0.316 onto ArcFace runs
too, and that does not fail - it silently mis-scores every
comparison."
  (dolist (option '(facecomp-detector-model
                    facecomp-encoder-model
                    facecomp-backend
                    facecomp-arcface-model
                    facecomp-threshold
                    facecomp-max
                    facecomp-detection-confidence))
    (should (null (eval (car (get option 'standard-value)) t)))))

(ert-deftest facecomp-test-the-executable-defaults-to-a-bare-name-on-path ()
  "Not an absolute path: the AppImage can be installed anywhere."
  (should (equal (eval (car (get 'facecomp-executable 'standard-value)) t)
                 "facecomp")))

(ert-deftest facecomp-test-the-entry-point-is-autoloaded ()
  "`facecomp-compare' has to be reachable before the package is loaded.

It is the only autoloaded symbol in the file, and without the cookie
\\[execute-extended-command] cannot find it at all until something
else has pulled `facecomp' in."
  (let ((source (with-temp-buffer
                  (insert-file-contents
                   (expand-file-name "facecomp.el" facecomp-test--directory))
                  (buffer-string))))
    (should (string-match-p ";;;###autoload\n(defun facecomp-compare " source))))

;;;; facecomp--check-ranges

(ert-deftest facecomp-test-unset-options-pass-the-range-check ()
  "Every range-checked variable is nil-able, and nil must not be range-checked.

This is the default configuration - the recommended one - so an
over-eager check here would reject the setup most runs use."
  (facecomp-test--with-defaults
    (should-not (facecomp--check-ranges nil))))

(ert-deftest facecomp-test-a-threshold-of-one-is-rejected ()
  "It is a plausible thing to type and it used to report a face as 0.0%.

Cosine similarity never reaches 1.0 for two separate photographs, so
the cutoff is exclusive at both ends."
  (facecomp-test--with-defaults
    (let ((facecomp-threshold 1.0))
      (should-error (facecomp--check-ranges nil) :type 'user-error))
    (let ((facecomp-threshold 0.0))
      (should-error (facecomp--check-ranges nil) :type 'user-error))
    (let ((facecomp-threshold "0.5"))
      (should-error (facecomp--check-ranges nil) :type 'user-error))
    (let ((facecomp-threshold 0.316))
      (should-not (facecomp--check-ranges nil)))))

(ert-deftest facecomp-test-a-detector-confidence-of-one-is-allowed ()
  "The two ranges are deliberately not the same, and must not be unified.

A threshold of 1.0 is meaningless, but a detector confidence of 1.0 is
merely the strictest possible filter - \"only perfectly certain faces\"
- which is a coherent thing to ask for."
  (facecomp-test--with-defaults
    (should-not (facecomp--check-ranges 1.0))
    (should-error (facecomp--check-ranges 0.0) :type 'user-error)
    (should-error (facecomp--check-ranges 1.5) :type 'user-error)
    (should-error (facecomp--check-ranges "0.8") :type 'user-error)))

(ert-deftest facecomp-test-max-must-be-a-positive-whole-number ()
  "`--max' truncates an already-sorted list, so a fraction has no meaning."
  (facecomp-test--with-defaults
    (let ((facecomp-max 0))
      (should-error (facecomp--check-ranges nil) :type 'user-error))
    (let ((facecomp-max -1))
      (should-error (facecomp--check-ranges nil) :type 'user-error))
    (let ((facecomp-max 2.5))
      (should-error (facecomp--check-ranges nil) :type 'user-error))
    (let ((facecomp-max 5))
      (should-not (facecomp--check-ranges nil)))))

(ert-deftest facecomp-test-the-range-check-names-the-variable-at-fault ()
  "The whole point of checking here rather than letting the executable
refuse is that the message says which setting is wrong."
  (facecomp-test--with-defaults
    (let ((facecomp-threshold 1.0))
      (should (string-match-p "facecomp-threshold"
                              (cadr (should-error (facecomp--check-ranges nil)
                                                  :type 'user-error)))))
    (let ((facecomp-max 0))
      (should (string-match-p "facecomp-max"
                              (cadr (should-error (facecomp--check-ranges nil)
                                                  :type 'user-error)))))))

;;;; facecomp--model-args

(ert-deftest facecomp-test-unset-model-paths-produce-no-flags ()
  "The AppImage bakes its own model paths in; passing nothing uses them."
  (facecomp-test--with-defaults
    (should (null (facecomp--model-args)))))

(ert-deftest facecomp-test-a-model-path-that-points-nowhere-is-caught-early ()
  "Set-but-missing is a typo, not a request to fall back to the default.

Silently omitting the flag would run against the AppImage's bundled
weights while the user believed their own were in use."
  (facecomp-test--with-defaults
    (let ((facecomp-detector-model "/nonexistent/yunet.onnx"))
      (should-error (facecomp--model-args) :type 'user-error))
    (let ((facecomp-encoder-model "/nonexistent/sface.onnx"))
      (should-error (facecomp--model-args) :type 'user-error))
    (let ((facecomp-backend 'arcface)
          (facecomp-arcface-model "/nonexistent/arcface.onnx"))
      (should-error (facecomp--model-args) :type 'user-error))))

(ert-deftest facecomp-test-the-backend-decides-which-weights-are-sent ()
  "The weights can never disagree with the backend they are loaded for.

`--encoder-model' stays the flag name in both cases; what changes is
which customize variable supplies it.  Sending SFace's 128-dimension
weights to an ArcFace run is not a load error waiting to happen, it is
a different embedding space and therefore a different scale of
similarity, against a cutoff derived for neither."
  (facecomp-test--with-temp-files (sface arcface)
    (facecomp-test--with-defaults
      (let ((facecomp-encoder-model sface)
            (facecomp-arcface-model arcface))
        (should (equal (facecomp--model-args) (list "--encoder-model" sface)))
        (let ((facecomp-backend 'arcface))
          (should (equal (facecomp--model-args)
                         (list "--encoder-model" arcface))))
        (let ((facecomp-backend 'sface))
          (should (equal (facecomp--model-args)
                         (list "--encoder-model" sface))))))))

(ert-deftest facecomp-test-an-arcface-run-without-arcface-weights-passes-none ()
  "Falling through to `facecomp-encoder-model' here would hand SFace's
weights to ArcFace.  When only the SFace path is set and the backend
is ArcFace, the flag is omitted so the executable uses its own bundled
ArcFace weights instead."
  (facecomp-test--with-temp-files (sface)
    (facecomp-test--with-defaults
      (let ((facecomp-encoder-model sface)
            (facecomp-backend 'arcface))
        (should (null (facecomp--model-args)))))))

(ert-deftest facecomp-test-detector-and-encoder-flags-are-independent ()
  "Either can be set without the other."
  (facecomp-test--with-temp-files (detector encoder)
    (facecomp-test--with-defaults
      (let ((facecomp-detector-model detector))
        (should (equal (facecomp--model-args)
                       (list "--detector-model" detector))))
      (let ((facecomp-detector-model detector)
            (facecomp-encoder-model encoder))
        (should (equal (facecomp--model-args)
                       (list "--detector-model" detector
                             "--encoder-model" encoder)))))))

;;;; facecomp--parse-report

(ert-deftest facecomp-test-a-report-is-parsed-past-leading-noise ()
  "The AppImage prints its FUSE diagnostic to stdout, ahead of the payload.

The run succeeds and the report is correct; the line just sits in
front of it.  Anchoring to the first brace keeps that from failing an
otherwise good comparison."
  (let ((report (facecomp--parse-report
                 (concat "No suitable fusermount binary found on the $PATH\n"
                         "{\"threshold\": 0.316}"))))
    (should (equal (alist-get 'threshold report) 0.316))))

(ert-deftest facecomp-test-output-with-no-json-object-is-an-error ()
  "Better a named failure than an empty report rendered as a result."
  (should-error (facecomp--parse-report "error: no face detected\n")))

(ert-deftest facecomp-test-a-report-parses-to-alists-and-lists ()
  "`facecomp--render' walks the report with `alist-get' and `dolist'.

Parsing to hash tables or vectors instead would not error here - it
would error, or silently render nothing, only once a real comparison
came back."
  (let* ((report (facecomp--parse-report
                  "{\"results\": [{\"photo\": \"a.jpg\", \"match_percent\": 91.25}]}"))
         (results (alist-get 'results report)))
    (should (listp results))
    (should (proper-list-p results))
    (should (equal (alist-get 'photo (car results)) "a.jpg"))))

;;;; facecomp--percent-face

(ert-deftest facecomp-test-the-confidence-colours-change-at-eighty-and-forty-five ()
  "These are the boundaries the CONFIDENCE LABELS help section describes."
  (should (eq (facecomp--percent-face 100) 'success))
  (should (eq (facecomp--percent-face 80) 'success))
  (should (eq (facecomp--percent-face 79.9) 'warning))
  (should (eq (facecomp--percent-face 45) 'warning))
  (should (eq (facecomp--percent-face 44.9) 'error))
  (should (eq (facecomp--percent-face 0) 'error)))

;;;; facecomp--run

(ert-deftest facecomp-test-a-missing-executable-is-reported-by-name ()
  "Without this the failure is a raw subprocess error naming nothing useful."
  (facecomp-test--with-defaults
    (let ((facecomp-executable "facecomp-does-not-exist"))
      (should (string-match-p
               "facecomp-does-not-exist"
               (cadr (should-error (facecomp--run '("/m.jpg") '("/t.jpg"))
                                   :type 'user-error)))))))

(ert-deftest facecomp-test-a-flag-always-separates-the-masters-from-the-targets ()
  "`--master' takes one or more values, so it swallows everything until
the next flag.  `--json' and `--slave' are unconditional and sit
between the two lists precisely so the targets can never be absorbed
into the master set - which would not fail, it would compare the
person against themselves and report a near-perfect match for every
photo."
  (facecomp-test--with-defaults
    (facecomp-test--with-stub-process "{\"results\": []}" 0
      (facecomp--run '("/m.jpg") '("/t1.jpg" "/t2.jpg"))
      (should (equal facecomp-test--process-args
                     '("--master" "/m.jpg" "--json" "--slave"
                       "/t1.jpg" "/t2.jpg")))
      (let ((master-at (cl-position "--master" facecomp-test--process-args
                                    :test #'string=))
            (first-target (cl-position "/t1.jpg" facecomp-test--process-args
                                       :test #'string=)))
        (should (cl-find-if (lambda (a) (string-prefix-p "--" a))
                            (cl-subseq facecomp-test--process-args
                                       (1+ master-at) first-target)))))))

(ert-deftest facecomp-test-every-configured-option-reaches-the-command-line ()
  "The whole argument vector, in order, for a fully configured run."
  (facecomp-test--with-temp-files (detector arcface)
    (facecomp-test--with-defaults
      (let ((facecomp-detector-model detector)
            (facecomp-arcface-model arcface)
            (facecomp-backend 'arcface)
            (facecomp-threshold 0.4)
            (facecomp-detection-confidence 0.7)
            (facecomp-max 3))
        (facecomp-test--with-stub-process "{\"results\": []}" 0
          (facecomp--run '("/m1.jpg" "/m2.jpg") '("/t.jpg"))
          (should (equal facecomp-test--process-args
                         (list "--master" "/m1.jpg" "/m2.jpg"
                               "--detector-model" detector
                               "--encoder-model" arcface
                               "--backend" "arcface"
                               "--threshold" "0.4"
                               "--detection-confidence" "0.7"
                               "--max" "3"
                               "--json" "--slave"
                               "/t.jpg"))))))))

(ert-deftest facecomp-test-a-bare-master-string-still-works ()
  "Older callers passed one photo rather than a list."
  (facecomp-test--with-defaults
    (facecomp-test--with-stub-process "{\"results\": []}" 0
      (facecomp--run "/m.jpg" '("/t.jpg"))
      (should (equal facecomp-test--process-args
                     '("--master" "/m.jpg" "--json" "--slave" "/t.jpg"))))))

(ert-deftest facecomp-test-a-one-off-confidence-does-not-touch-the-setting ()
  "The prefix-argument prompt is for re-checking one borderline result
at a stricter setting.  Writing it back to the customize variable
would silently change every later run too."
  (facecomp-test--with-defaults
    (let ((facecomp-detection-confidence 0.5))
      (facecomp-test--with-stub-process "{\"results\": []}" 0
        (facecomp--run '("/m.jpg") '("/t.jpg") 0.9)
        (should (member "0.9" facecomp-test--process-args))
        (should-not (member "0.5" facecomp-test--process-args)))
      (should (equal facecomp-detection-confidence 0.5)))))

(ert-deftest facecomp-test-a-bad-setting-stops-the-run-before-it-starts ()
  "No subprocess at all: the check is there to replace the error dump."
  (facecomp-test--with-defaults
    (let ((facecomp-threshold 1.0))
      (facecomp-test--with-stub-process "{\"results\": []}" 0
        (should-error (facecomp--run '("/m.jpg") '("/t.jpg")) :type 'user-error)
        (should (null facecomp-test--process-args))))
    (facecomp-test--with-stub-process "{\"results\": []}" 0
      (should-error (facecomp--run '("/m.jpg") '("/t.jpg") 1.5) :type 'user-error)
      (should (null facecomp-test--process-args)))))

(ert-deftest facecomp-test-a-failed-run-reports-status-and-stderr ()
  "Unparseable output means the run failed, and what it said is the
only thing that explains why.  Merging stderr into the buffer being
parsed would corrupt good runs, so it is held aside and re-attached
here."
  (facecomp-test--with-defaults
    (cl-letf (((symbol-function 'executable-find) (lambda (&rest _) "/bin/false"))
              ((symbol-function 'call-process)
               (lambda (_program _infile destination _display &rest _args)
                 (insert "not json at all")
                 (write-region "error: could not open /m.jpg" nil
                               (cadr destination) nil 'silent)
                 2)))
      (let ((message (cadr (should-error (facecomp--run '("/m.jpg") '("/t.jpg"))
                                         :type 'error))))
        (should (string-match-p "status 2" message))
        (should (string-match-p "not json at all" message))
        (should (string-match-p "could not open /m.jpg" message))))))

;;;; facecomp--render

(ert-deftest facecomp-test-several-masters-are-shown-as-one-template ()
  "The reader has to be able to see that averaging happened, and over what."
  (let ((out (facecomp-test--render
              '((masters . ("/a.jpg" "/b.jpg" "/c.jpg")) (results . nil)))))
    (should (string-match-p "master: 3 photos averaged into one template" out))
    (dolist (photo '("/a.jpg" "/b.jpg" "/c.jpg"))
      (should (string-match-p (regexp-quote photo) out)))))

(ert-deftest facecomp-test-a-single-master-is-shown-plainly ()
  "One photo is not a template, and saying \"1 photo averaged\" would
imply the more permissive scoring that averaging actually causes."
  (let ((out (facecomp-test--render '((masters . ("/a.jpg")) (results . nil)))))
    (should (string-match-p "master: /a.jpg" out))
    (should-not (string-match-p "averaged" out))))

(ert-deftest facecomp-test-an-older-executables-single-master-key-still-renders ()
  "Builds before centroid matching report `master', not `masters'.

Handling only the new key would give those runs an empty header rather
than an error, which is the kind of thing nobody reports."
  (let ((out (facecomp-test--render '((master . "/a.jpg") (results . nil)))))
    (should (string-match-p "master: /a.jpg" out))))

(ert-deftest facecomp-test-masters-arriving-as-a-vector-still-render ()
  "`append ... nil' is what makes that so, and it is easy to drop as noise."
  (let ((out (facecomp-test--render
              '((masters . ["/a.jpg" "/b.jpg"]) (results . nil)))))
    (should (string-match-p "master: 2 photos averaged into one template" out))))

(ert-deftest facecomp-test-enrolment-disagreement-is-warned-about ()
  "Nothing else catches a stray photo of the wrong person among the masters.

It drags the template toward them and then mis-scores every
comparison, silently; the lowest pairwise similarity is the only
visible symptom.  The comparison is strictly below the cutoff, so a
set that agrees exactly at it is not flagged."
  ;; Case-sensitively: the no-results line says "see warnings above", and
  ;; `string-match-p' folds case unless told not to.
  (let* ((case-fold-search nil)
         (below (facecomp-test--render
                 '((masters . ("/a.jpg" "/b.jpg")) (enrolment_agreement . 0.21)
                   (threshold . 0.316) (results . nil))))
         (at (facecomp-test--render
              '((masters . ("/a.jpg" "/b.jpg")) (enrolment_agreement . 0.316)
                (threshold . 0.316) (results . nil)))))
    (should (string-match-p "agreement: 0.2100" below))
    (should (string-match-p "WARNING: below the 0.316 cutoff" below))
    (should (eq (facecomp-test--face-at below "WARNING") 'warning))
    (should (string-match-p "agreement: 0.3160" at))
    (should-not (string-match-p "WARNING" at))))

(ert-deftest facecomp-test-agreement-is-shown-even-with-no-threshold-to-judge-it ()
  "A template built for a backend whose cutoff the report omits still
gets its agreement figure; only the WARNING line needs the threshold."
  (let ((case-fold-search nil)
        (out (facecomp-test--render
              '((masters . ("/a.jpg" "/b.jpg")) (enrolment_agreement . 0.21)
                (results . nil)))))
    (should (string-match-p "agreement: 0.2100" out))
    (should-not (string-match-p "WARNING" out))))

(ert-deftest facecomp-test-a-one-photo-master-shows-no-agreement-line ()
  "There is no pair to disagree, and a figure would invite reading one."
  (let ((out (facecomp-test--render '((masters . ("/a.jpg")) (results . nil)))))
    (should-not (string-match-p "agreement" out))))

(ert-deftest facecomp-test-the-embedding-width-is-shown-when-reported ()
  "128 or 512 is how the reader tells which backend actually ran.
Builds that predate the ArcFace backend do not report it at all."
  (should (string-match-p
           "embedding: 512 dimensions per face"
           (facecomp-test--render '((embedding_dimensions . 512) (results . nil)))))
  (should-not (string-match-p
               "embedding:"
               (facecomp-test--render '((results . nil))))))

(ert-deftest facecomp-test-results-carry-their-percentage-and-label ()
  "The percentage is coloured by the same boundaries the labels use."
  (let ((out (facecomp-test--render
              '((masters . ("/a.jpg"))
                (results . (((photo . "/t1.jpg") (match_percent . 91.25)
                             (confidence . "Almost certain"))
                            ((photo . "/t2.jpg") (match_percent . 12.0)
                             (confidence . "Almost no chance"))))))))
    (should (string-match-p "/t1.jpg" out))
    (should (string-match-p "91.2% match" out))
    (should (string-match-p "Almost certain" out))
    (should (eq (facecomp-test--face-at out "91.2%") 'success))
    (should (eq (facecomp-test--face-at out "12.0%") 'error))))

(ert-deftest facecomp-test-a-photo-with-several-faces-says-so ()
  "Only the best-matching face is scored, and the reader needs to know
that a choice was made.  One face is the ordinary case and is silent."
  (let ((many (facecomp-test--render
               '((results . (((photo . "/t.jpg") (match_percent . 50.0)
                              (confidence . "Likely") (faces_detected . 4)))))))
        (one (facecomp-test--render
              '((results . (((photo . "/t.jpg") (match_percent . 50.0)
                             (confidence . "Likely") (faces_detected . 1))))))))
    (should (string-match-p "\\[best of 4 faces in photo\\]" many))
    (should-not (string-match-p "best of" one))))

(ert-deftest facecomp-test-per-photo-errors-are-shown-before-the-results ()
  "A photo that yielded no face is absent from the results entirely, so
the warning is the only trace of it."
  (let ((out (facecomp-test--render
              '((errors . ("/bad.jpg: no face detected"))
                (results . (((photo . "/t.jpg") (match_percent . 50.0)
                             (confidence . "Likely"))))))))
    (should (string-match-p "warning: /bad.jpg: no face detected" out))
    (should (eq (facecomp-test--face-at out "warning:") 'warning))
    (should (< (string-match "warning:" out) (string-match "/t.jpg" out)))))

(ert-deftest facecomp-test-no-results-points-at-the-warnings ()
  "An empty buffer reads as a clean run that found nothing to say."
  (let ((out (facecomp-test--render
              '((errors . ("/bad.jpg: no face detected")) (results . nil)))))
    (should (string-match-p "No comparable photos (see warnings above)" out))))

(ert-deftest facecomp-test-a-second-run-replaces-the-first ()
  "`special-mode' leaves the buffer read-only, so re-rendering into it
needs `inhibit-read-only'.  Without it the second comparison of a
session fails outright."
  (facecomp-test--render '((masters . ("/first.jpg")) (results . nil)))
  ;; Deliberately not through `facecomp-test--render', which kills the
  ;; buffer first: rendering into the read-only buffer the previous
  ;; comparison left behind is the entire thing being tested, and a fresh
  ;; buffer would never reach it.
  (facecomp--render '((masters . ("/second.jpg")) (results . nil)))
  (let ((out (with-current-buffer "*facecomp*" (buffer-string))))
    (should (string-match-p "/second.jpg" out))
    (should-not (string-match-p "/first.jpg" out)))
  (with-current-buffer "*facecomp*"
    (should (derived-mode-p 'special-mode))
    (should buffer-read-only)
    (should (= (point) (point-min)))))

;;;; facecomp--choose-masters

(ert-deftest facecomp-test-returning-nothing-takes-the-topmost-marked-file ()
  "Dired's marked-file list is buffer order, not the order they were
marked in - Dired does not track that - so RET alone has to mean
something positional and stated, rather than a guess."
  (cl-letf (((symbol-function 'completing-read-multiple) (lambda (&rest _) nil)))
    (should (equal (facecomp--choose-masters '("/d/a.jpg" "/d/b.jpg" "/d/c.jpg"))
                   '(("/d/a.jpg") ("/d/b.jpg" "/d/c.jpg"))))))

(ert-deftest facecomp-test-several-marked-files-can-be-masters ()
  "This is how a set of confirmed photos becomes one template from Dired.
Every one named has to come out of the targets, or the person would be
compared against their own enrolment photos."
  (cl-letf (((symbol-function 'completing-read-multiple)
             (lambda (&rest _) '("a.jpg" "c.jpg"))))
    (should (equal (facecomp--choose-masters '("/d/a.jpg" "/d/b.jpg" "/d/c.jpg"))
                   '(("/d/a.jpg" "/d/c.jpg") ("/d/b.jpg"))))))

(ert-deftest facecomp-test-targets-keep-their-dired-order ()
  "Results are sorted best-first by the executable, but the order photos
are handed over in is what breaks ties, so it has to be reproducible."
  (cl-letf (((symbol-function 'completing-read-multiple)
             (lambda (&rest _) '("b.jpg"))))
    (should (equal (cadr (facecomp--choose-masters
                          '("/d/a.jpg" "/d/b.jpg" "/d/c.jpg" "/d/d.jpg")))
                   '("/d/a.jpg" "/d/c.jpg" "/d/d.jpg")))))

(ert-deftest facecomp-test-a-name-that-was-not-marked-is-refused ()
  "Completion is required-match, but a name can still arrive by other
routes, and silently dropping it would compare against a set the user
did not choose."
  (cl-letf (((symbol-function 'completing-read-multiple)
             (lambda (&rest _) '("elsewhere.jpg"))))
    (should-error (facecomp--choose-masters '("/d/a.jpg" "/d/b.jpg"))
                  :type 'user-error)))

(ert-deftest facecomp-test-choosing-every-marked-file-leaves-no-targets ()
  "`facecomp-compare' refuses this rather than running a comparison
against nothing; what matters here is that the target list is empty
rather than accidentally still holding a master."
  (cl-letf (((symbol-function 'completing-read-multiple)
             (lambda (&rest _) '("a.jpg" "b.jpg"))))
    (should (equal (facecomp--choose-masters '("/d/a.jpg" "/d/b.jpg"))
                   '(("/d/a.jpg" "/d/b.jpg") nil)))))

;;;; The prompts

(ert-deftest facecomp-test-master-photos-come-back-in-the-order-chosen ()
  "They are accumulated with `push', so the list is reversed at the end."
  (let ((answers '("/d/one.jpg" "/d/two.jpg" "/d/three.jpg"))
        (asked 0))
    (cl-letf (((symbol-function 'read-file-name)
               (lambda (&rest _) (pop answers)))
              ((symbol-function 'y-or-n-p)
               (lambda (&rest _) (< (cl-incf asked) 3))))
      (should (equal (facecomp--read-masters)
                     '("/d/one.jpg" "/d/two.jpg" "/d/three.jpg"))))))

(ert-deftest facecomp-test-one-master-is-the-ordinary-case ()
  "Answering no at the first prompt gives a plain one-photo comparison."
  (cl-letf (((symbol-function 'read-file-name) (lambda (&rest _) "/d/one.jpg"))
            ((symbol-function 'y-or-n-p) (lambda (&rest _) nil)))
    (should (equal (facecomp--read-masters) '("/d/one.jpg")))))

(ert-deftest facecomp-test-a-tilde-in-a-chosen-path-is-expanded ()
  "A subprocess does not know what `~' means.

`read-file-name' hands back whatever was typed once it is absolute, and
\"~/photos/one.jpg\" counts as absolute to Emacs and to nothing else:
`call-process' would pass it through verbatim and the executable would
look for a directory literally named `~'.  Both prompts expand, so a
home-relative path works from either."
  (let ((home (expand-file-name "~/photos/one.jpg")))
    (should-not (string-match-p "~" home))
    (cl-letf (((symbol-function 'read-file-name)
               (lambda (&rest _) "~/photos/one.jpg"))
              ((symbol-function 'y-or-n-p) (lambda (&rest _) nil)))
      (should (equal (facecomp--read-masters) (list home)))
      (should (equal (facecomp--read-targets "/photos/master.jpg")
                     (list home))))))

(ert-deftest facecomp-test-target-picking-starts-in-the-masters-directory ()
  "That is where the photos being compared almost always live, rather
than wherever \\[execute-extended-command] happened to be invoked from."
  (let ((dirs nil)
        (default-directory "/somewhere/else/"))
    (cl-letf (((symbol-function 'read-file-name)
               (lambda (_prompt &optional dir &rest _)
                 (push dir dirs)
                 "/photos/t.jpg"))
              ((symbol-function 'y-or-n-p) (lambda (&rest _) nil)))
      (facecomp--read-targets "/photos/master.jpg")
      (should (equal dirs '("/photos/")))
      (setq dirs nil)
      (facecomp--read-targets nil)
      (should (equal dirs '("/somewhere/else/"))))))

(ert-deftest facecomp-test-a-one-off-confidence-is-checked-as-it-is-typed ()
  "Same range as the customize variable, refused at the prompt so the
user can retype it rather than having the run die."
  (cl-letf (((symbol-function 'read-number) (lambda (&rest _) 1.5)))
    (should-error (facecomp--read-confidence) :type 'user-error))
  (cl-letf (((symbol-function 'read-number) (lambda (&rest _) 0.0)))
    (should-error (facecomp--read-confidence) :type 'user-error))
  (cl-letf (((symbol-function 'read-number) (lambda (&rest _) 1.0)))
    (should (equal (facecomp--read-confidence) 1.0))))

(ert-deftest facecomp-test-the-confidence-prompt-defaults-to-nought-point-eight ()
  "The value the help documents for when every result must be trustworthy,
which is the usual reason to re-run a borderline comparison at all."
  (let (default)
    (cl-letf (((symbol-function 'read-number)
               (lambda (_prompt &optional def &rest _) (setq default def) 0.8)))
      (facecomp--read-confidence)
      (should (equal default 0.8)))))

;;;; facecomp-compare

(ert-deftest facecomp-test-a-comparison-needs-a-master-and-a-target ()
  "Both lists can come back empty from Dired - every marked file named
as a master leaves no targets - and an empty `--master' would make the
executable's own error the first sign of it."
  ;; With the subprocess stubbed, a missing guard does not fail somewhere
  ;; else - it succeeds. An earlier version of this test asserted only the
  ;; error type and passed happily with both guards deleted, because
  ;; `executable-find' further down raised a `user-error' of its own.
  (facecomp-test--with-defaults
    (facecomp-test--with-stub-process "{\"results\": []}" 0
      (should (string-match-p
               "at least one master photo"
               (cadr (should-error (facecomp-compare nil '("/t.jpg"))
                                   :type 'user-error))))
      (should (string-match-p
               "at least one photo to compare"
               (cadr (should-error (facecomp-compare '("/m.jpg") nil)
                                   :type 'user-error))))
      (should (null facecomp-test--process-args)))))

(ert-deftest facecomp-test-a-comparison-runs-and-renders ()
  "The whole path, from arguments to buffer, with the subprocess stubbed."
  (facecomp-test--with-defaults
    (when (get-buffer "*facecomp*") (kill-buffer "*facecomp*"))
    (facecomp-test--with-stub-process
        (concat "{\"masters\": [\"/m.jpg\"], \"threshold\": 0.316,"
                " \"embedding_dimensions\": 128,"
                " \"results\": [{\"photo\": \"/t.jpg\", \"match_percent\": 88.5,"
                " \"confidence\": \"Very likely\", \"faces_detected\": 1}]}")
        0
      (facecomp-compare '("/m.jpg") '("/t.jpg"))
      (should (equal facecomp-test--process-args
                     '("--master" "/m.jpg" "--json" "--slave" "/t.jpg")))
      (let ((out (with-current-buffer "*facecomp*" (buffer-string))))
        (should (string-match-p "master: /m.jpg" out))
        (should (string-match-p "embedding: 128 dimensions per face" out))
        (should (string-match-p "88.5% match" out))
        (should (string-match-p "Very likely" out))))))

(provide 'facecomp-tests)

;;; facecomp-tests.el ends here
