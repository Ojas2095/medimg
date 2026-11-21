PY ?= python
export PYTHONPATH := src

.PHONY: install test fundamentals data baseline gui infer evaluate robustness gradcam benchmarks verify clean

install:        ## pip install -e . with all optional dependencies
	$(PY) -m pip install -e ".[all]"

test:           ## unit tests (NumPy fundamentals, vision models, pipeline, UI, checkpoints)
	$(PY) -m pytest -v

fundamentals:   ## steps 3-6: all demos + ablations -> out/figures, out/fundamentals_results.json
	$(PY) -m medimg fundamentals all --out out

data:           ## synthetic data + dedup + leakage-safe split -> data/synth
	$(PY) -m medimg data --root data/synth --n 300

baseline: test fundamentals data   ## everything that runs without torch
	@echo "BASELINE OK"

gui:            ## launch interactive Gradio workstation UI on http://127.0.0.1:7860
	$(PY) -m medimg gui --port 7860

infer:          ## run single-image inference pipeline: make infer IMAGE=path/to/img.jpg OUT=out/triage
	$(PY) -m medimg infer --image $(IMAGE) --out $(OUT)

evaluate:       ## run full quantitative evaluation on held-out test split
	$(PY) -m medimg evaluate

robustness:     ## run clinical robustness stress-testing suite
	$(PY) -m medimg robustness

gradcam:        ## run Grad-CAM visual interpretability & error audit
	$(PY) -m medimg gradcam

benchmarks:     ## execute the experimental benchmark matrix across all models
	$(PY) scripts/run_benchmarks.py

verify:         ## execute automated aggressive model & pipeline stress-testing suite
	$(PY) scripts/aggressive_model_check.py

clean:
	rm -rf out data runs/temp build *.egg-info .pytest_cache
