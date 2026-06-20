PYTHON_VERSION ?= 3.12
DATASET ?= shayanfazeli/heartbeat
DATA_DIR ?= data/raw
RESULTS_DIR ?= results
BOTTLENECKS ?= 2 4 8 16 32 64 96
EPOCHS ?= 80
BATCH_SIZE ?= 2048
NUM_WORKERS ?= 0
HIDDEN_DIMS ?= 2048 1024 512 256
DEVICE ?= cuda
AMP ?= 1
LIMIT ?=
INFER_K ?= 8
BEST_METRIC ?= prd
INFER_INDICES ?=
INFER_SAMPLES ?= 6

export LD_LIBRARY_PATH := /run/opengl-driver/lib:$(LD_LIBRARY_PATH)

.PHONY: help sync download-data train evaluate infer demo quick-demo test clean-results

help:
	@echo "Targets:"
	@echo "  make sync           Install dependencies with uv"
	@echo "  make download-data  Download Kaggle heartbeat dataset to $(DATA_DIR)"
	@echo "  make train          Train MLP autoencoders on $(DEVICE)"
	@echo "  make evaluate       Evaluate models and generate plots"
	@echo "  make infer          Run inference for checkpoint K=$(INFER_K)"
	@echo "  make demo           Run inference for the best existing model by $(BEST_METRIC)"
	@echo "  make quick-demo     Train K=$(INFER_K) briefly, evaluate, and infer"
	@echo "  make test           Run tests"
	@echo "  make clean-results  Remove generated results"

sync:
	uv sync --python $(PYTHON_VERSION)

download-data: sync
	@mkdir -p $(DATA_DIR)
	uv run kaggle datasets download -d $(DATASET) -p $(DATA_DIR) --unzip
	@test -f $(DATA_DIR)/mitbih_train.csv
	@test -f $(DATA_DIR)/mitbih_test.csv
	@echo "Downloaded MIT-BIH CSV files to $(DATA_DIR)"

train: sync
	uv run ecg-train --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR) --bottlenecks $(BOTTLENECKS) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) --hidden-dims $(HIDDEN_DIMS) --device $(DEVICE) $(if $(filter 1 true yes,$(AMP)),--amp,) $(if $(LIMIT),--limit $(LIMIT),)

evaluate: sync
	uv run ecg-evaluate --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR) $(if $(LIMIT),--limit $(LIMIT),)

infer: sync
	uv run ecg-infer --checkpoint $(RESULTS_DIR)/models/mlp_autoencoder_k$(INFER_K).pt --input-csv $(DATA_DIR)/mitbih_test.csv --output-dir $(RESULTS_DIR)/inference_k$(INFER_K) $(if $(INFER_INDICES),--indices $(INFER_INDICES),--num-samples $(INFER_SAMPLES))

demo: sync
	$(eval BEST_CHECKPOINT := $(shell uv run ecg-best-model --results-dir $(RESULTS_DIR) --metric $(BEST_METRIC)))
	uv run ecg-infer --checkpoint $(BEST_CHECKPOINT) --input-csv $(DATA_DIR)/mitbih_test.csv --output-dir $(RESULTS_DIR)/demo_best_$(BEST_METRIC) $(if $(INFER_INDICES),--indices $(INFER_INDICES),--num-samples $(INFER_SAMPLES))

quick-demo: sync
	uv run ecg-train --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR) --bottlenecks $(INFER_K) --epochs 5 --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) --hidden-dims $(HIDDEN_DIMS) --device $(DEVICE) $(if $(filter 1 true yes,$(AMP)),--amp,) --limit 4096
	uv run ecg-evaluate --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR) --bottlenecks $(INFER_K) --limit 4096
	uv run ecg-infer --checkpoint $(RESULTS_DIR)/models/mlp_autoencoder_k$(INFER_K).pt --input-csv $(DATA_DIR)/mitbih_test.csv --output-dir $(RESULTS_DIR)/demo_inference_k$(INFER_K) --num-samples $(INFER_SAMPLES)

test: sync
	uv run pytest

clean-results:
	rm -rf $(RESULTS_DIR)
