# PreRiskBench
# PreRiskBench: A Benchmark for Preoperative Anesthesia Risk Assessment

PreRiskBench is a research framework for evaluating artificial intelligence models on preoperative anesthesia risk assessment. It organizes heterogeneous perioperative data into task-specific inputs, supports multiple modeling approaches, and provides evaluation tools for outcomes spanning preoperative physical status, intraoperative events, postoperative complications, and healthcare utilization.

The framework brings together structured clinical variables and preoperative clinical text to support comparisons among conventional machine learning, deep learning, pretrained language models, and large language models (LLMs).

## Overview

Preoperative assessment requires integrating information about a patient’s medical history, laboratory findings, planned surgery, and anesthesia management. Differences in data schemas, clinical terminology, available outcomes, and patient populations make it difficult to compare models across perioperative datasets.

PreRiskBench provides a common workflow for:

- Defining dataset-specific features and prediction targets.
- Preparing structured inputs and clinical summaries.
- Constructing task-specific benchmark samples.
- Training and evaluating predictive models.
- Evaluating LLMs through zero-shot prompting, retrieval-augmented generation (RAG), and fine-tuning workflows.
- Comparing performance across datasets, tasks, and modeling approaches.

Task coverage varies by dataset. A shared task name does not imply identical populations, label distributions, or outcome ascertainment across resources.

## Data Resources

The project includes feature and label configurations for the following resources:

| Resource | Configuration | Examples of configured outcomes |
|---|---|---|
| VitalDB | `VitalDB.yaml` | ASA physical status, blood loss, intraoperative hypotension, hypothermia, hypoxemia, acute kidney injury, ICU admission, length of stay, and in-hospital mortality |
| MOVER | `MOVER.yaml` | ASA physical status, ICU admission, and hospital length of stay |
| INSPIRE | `INSPIRE.yaml` | ASA physical status, intraoperative hypothermia, acute kidney injury, ICU and hospital length of stay, and in-hospital mortality |
| PreARiD | `PreARiD.yaml` | ASA physical status, blood loss, intraoperative events, postoperative symptoms and complications, and ICU admission |

Configurations are located in `config/feature_label_config/`. They describe available fields and labels; inclusion in a configuration does not guarantee that every task is enabled in every experiment.

Users must obtain access to each underlying resource separately. The presence of a dataset adapter does not grant permission to access or redistribute that dataset.

## Benchmark Tasks

PreRiskBench organizes prediction targets into five clinical categories.

| Category | Representative targets | Prediction formulations |
|---|---|---|
| Preoperative physical status | American Society of Anesthesiologists (ASA) physical status classification | Ordered classification |
| Intraoperative events | Estimated blood loss, severe blood loss, hypotension, hypothermia, and hypoxemia | Regression, ordered classification, or binary classification |
| Postoperative complications | Acute kidney injury and its stage; pain, nausea, vomiting, sore throat, pruritus, dizziness, sleep disturbance, and gastrointestinal, urinary, or respiratory dysfunction | Binary or ordered classification |
| Healthcare utilization | ICU admission, ICU length of stay, hospital length of stay, and prolonged stays | Binary classification, regression, or ordered classification |
| Mortality | In-hospital mortality | Binary classification |

Some clinical outcomes have several formulations. For example, length of stay may be represented as a continuous duration, a duration category, or a prolonged-stay indicator. These formulations are separate tasks and should be reported accordingly.

Dataset-specific definitions, thresholds, units, and category mappings are specified in the corresponding configuration files.

## Input Features

Depending on the resource, model inputs may include:

- **Patient demographics:** age, sex, height, weight, and body mass index.
- **Medical history:** organ-system conditions, previous surgery, and relevant history details.
- **Preoperative investigations:** laboratory measurements, abnormality indicators, and available examination findings.
- **Surgical information:** preoperative diagnoses, planned procedures, and surgical department.
- **Anesthesia information:** planned anesthesia type and available assessment variables.
- **Medication information:** preoperative medication records or summaries.
- **Additional assessment findings:** physical examination, airway assessment, consultation information, and preoperative notes where available.

The framework supports both structured features for predictive models and clinical summaries for language-model evaluation.

For preoperative prediction, input construction must respect the prediction time point. Intraoperative and postoperative information used to define outcomes must not be included as preoperative predictors.

## Modeling Approaches

| Approach | Implementations and workflows | Location |
|---|---|---|
| Conventional machine learning | Logistic regression, random forests, and XGBoost | `model_train_ml/` |
| Deep learning | Multilayer perceptron and Transformer models | `model_train_dl/` |
| Pretrained language models | BERT-based data preparation, training, and evaluation | `model_train_bert/` |
| LLM zero-shot inference | Prompt construction, inference, output validation, and metric aggregation | `model_infer_zeroshot/` |
| Retrieval-augmented LLM inference | Retrieval-data preparation, index construction, inference, and evaluation | `model_infer_rag/` |
| LLM fine-tuning | Data preparation and LLaMA-Factory-based fine-tuning workflows | `model_train_llm_finetuning/` |

Model availability, required dependencies, and supported task formulations differ between modules. Experiments should record the model checkpoint, feature representation, prompt configuration, and training or inference settings used.

## Repository Structure

```text
PreRiskBench/
├── config/
│   ├── feature_label_config/       # Dataset-specific feature and label definitions
│   └── datasets_config/            # Task and dataset configuration
├── data_pipeline/
│   └── benchmark_samples/          # Benchmark sampling, export, and distribution analysis
├── model_train_ml/                # Conventional machine-learning baselines
├── model_train_dl/                # Deep-learning baselines
├── model_train_bert/              # BERT-based modeling
├── model_train_llm_finetuning/    # LLM fine-tuning workflows
├── model_infer_zeroshot/          # Zero-shot LLM evaluation
├── model_infer_rag/               # Retrieval-augmented LLM evaluation
├── runners/                      # Training and evaluation entry points
├── src/
│   ├── models/                   # Shared model interfaces
│   ├── runners/                  # Shared execution logic
│   ├── utils/                    # Data, prompting, splitting, and metric utilities
│   └── analysis/                 # Prediction and result analysis
└── requirements.txt              # Base Python dependencies
```

## Getting Started

### 1. Install the base dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Individual workflows may require additional dependencies, local model checkpoints, database drivers, or a compatible inference service. Install these according to the module you intend to use.

### 2. Obtain and configure data

Obtain authorized access to the selected datasets and prepare their local feature and label tables.

Review:

- `config/feature_label_config/` for feature selection, label definitions, and source-table mappings.
- `config/datasets_config/` for task and dataset settings.
- The configuration files within the selected modeling module.

Set database connections, input paths, and model locations for your own environment. Keep credentials and private endpoints outside version control.

### 3. Prepare benchmark inputs

Use `data_pipeline/benchmark_samples/` to inspect the sampling and export workflow:

```bash
python data_pipeline/benchmark_samples/sample_benchmark.py --help
```

Prepare the inputs required by the selected model family using its data-preparation scripts. Structured models, BERT-based models, and LLM workflows use different input formats.

Before running experiments, verify:

- Available labels and task-specific eligibility.
- Missing values, category mappings, and measurement units.
- Training, validation, and test membership.
- Separation of predictor fields from outcome fields.
- Consistency between exported inputs and the selected task configuration.

### 4. Run a baseline

After preparing the module-specific data and configuring the local text-encoder checkpoint, an example XGBoost experiment for ASA classification is:

```bash
python model_train_ml/train_ml.py \
  --task ASA \
  --datasets vitaldb \
  --model_type xgboost \
  --mode both \
  --device cpu
```

Inspect available options with:

```bash
python model_train_ml/train_ml.py --help
python model_train_dl/train_dl.py --help
```

The training interfaces also expose dataset-selection options for experiments involving multiple resources. Cross-resource comparisons require compatible feature preparation and label definitions.

### 5. Evaluate language models

Choose the workflow appropriate to the experiment:

- `model_infer_zeroshot/` for direct prediction from clinical summaries.
- `model_infer_rag/` for prediction with retrieved examples.
- `model_train_llm_finetuning/` for supervised adaptation.

Configure the model backend and input paths before inference. Record prompt versions, decoding parameters, retrieval settings, and model identifiers alongside the results.

For RAG experiments, construct the retrieval collection from the permitted training or reference partition. Test cases and their outcomes must not enter the retrieval collection.

## Evaluation

Shared evaluation utilities are provided in `src/utils/metrics_utils.py`.

| Task type | Available evaluation measures |
|---|---|
| Classification | Accuracy, precision, sensitivity, specificity, Macro-F1, predictive values, Matthews correlation coefficient, and confusion matrices |
| Probabilistic classification | ROC-AUC, PR-AUC, Brier score, expected calibration error, and decision-curve analysis where applicable |
| Ordered classification | Classification metrics, mean absolute error, and weighted kappa |
| Regression | Mean absolute error, root mean squared error, coefficient of determination, and symmetric mean absolute percentage error |

Metric applicability depends on the task, observed classes, and model output. Probability-based metrics require appropriate probability estimates. Invalid or unparseable LLM outputs should be counted and reported rather than silently excluded.

For meaningful comparisons:

- Use consistent evaluation cases and label mappings.
- Keep preprocessing fitted on training data separate from evaluation data.
- Check that repeated patients do not cross partitions.
- Check for overlapping cases when combining related resources.
- Report dataset-specific results alongside pooled summaries.
- Record sample sizes, class distributions, random seeds, and uncertainty estimates where available.

## Semantic Alignment Extension

Related work extends the benchmark through semantic alignment of heterogeneous perioperative terminology across VitalDB, MOVER, and INSPIRE.

This extension compares three representations:

- **Field-Aligned:** shared feature fields with resource-specific clinical values.
- **Concept-Aligned:** clinical text mapped to standardized SNOMED CT concepts.
- **Concept-Clustered:** concept representations grouped at a coarser semantic granularity.

The extension examines shared concept space, cross-resource retrieval support, and downstream ASA classification. Its ASA experiments cover a subset of the broader PreRiskBench task framework.

The alignment pipeline is maintained separately from the core benchmark workflow. Using it requires the corresponding implementation, compatible data preparation, and authorized access to the terminology resources.

## Data Access and Responsible Use

Clinical data remain subject to their original access conditions and data-use agreements. Obtain source data through the relevant providers and process them within an authorized environment.

Public contributions must not contain patient records, clinical narratives, patient identifiers, private split files, database credentials, or restricted terminology assets. Sharing code does not transfer rights to the underlying clinical data.

PreRiskBench is intended for research and methodological evaluation. Benchmark performance alone does not establish suitability for clinical deployment.

## Contributing

Contributions are welcome in the following areas:

- Additional dataset adapters and documented outcome definitions.
- New baseline models and inference workflows.
- Improvements to preprocessing, output validation, and evaluation.
- Reproducibility checks and synthetic test cases.
- Documentation and configuration examples.

When reporting a problem, include the relevant module, task, software environment, and a minimal synthetic example. Use GitHub Issues to discuss substantial changes before opening a pull request.

## Acknowledgments

We thank the teams behind the underlying perioperative data resources and the open-source software used in this project. Please acknowledge and cite the original data-resource publications and any models or methods used in your experiments.
