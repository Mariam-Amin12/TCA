PYTHON=python
PIP=$(PYTHON) -m pip
.PHONY: all help install env extract_merge_data feature_transformation risk_param_tuning train evaluate inference clean

all: feature_transformation

help:
	@echo "Makefile targets:"
	@echo "  make all (default)         -> run feature_transformation"
	@echo "  make extract_merge_data    -> generate data/merged/multi_turn_data.csv"
	@echo "  make feature_transformation-> generate processed train/validation/test"
	@echo "  make risk_param_tuning     -> generate config/optimized_params_risk.json"
	@echo "  make train                 -> run training (src/model.py)"
	@echo "  make evaluate              -> run evaluation (src/evaluate.py)"
	@echo "  make inference             -> run inference (src/inference.py)"
	@echo "  make install               -> install project in editable mode"
	@echo "  make clean                 -> remove build artifacts"
\
extract_merge_data: data/merged/multi_turn_data.csv

data/merged/multi_turn_data.csv: src/extract_merge_data.py 
	$(PYTHON) src/extract_merge_data.py

feature_transformation: data/processed/train.csv data/processed/validation.csv data/processed/test.csv

data/processed/train.csv data/processed/validation.csv data/processed/test.csv: src/feature_transformation.py data/merged/multi_turn_data.csv config/optimized_params_risk.json
	$(PYTHON) src/feature_transformation.py

risk_param_tuning: config/optimized_params_risk.json

config/optimized_params_risk.json: src/risk_param_tune.py src/extract_merge_data.py
	$(PYTHON) src/risk_param_tune.py

train:data/processed/train.csv data/processed/validation.csv data/processed/test.csv
	$(PYTHON) src/model.py


clean:
	rm -rf __pycache__ .ipynb_checkpoints build dist *.egg-info
