# Machine Learning Classification of Eight Drug–Drug Interaction Events

**八类药物相互作用事件的机器学习分类** — a reproducible course project comparing linear SVM, random forest and XGBoost on eight event categories among drug pairs **already annotated as interacting**. It is an independent classical-ML study inspired by CSK-DDI, not a reproduction of its neural architecture.

> **Scope:** This model assigns one of eight recorded interaction-event types conditional on an existing interaction annotation. It does not determine whether a drug pair interacts, whether combined use is safe, or whether combined use is clinically effective. It is not a clinical decision tool.

## Data sources and scope

- DDI benchmark and SMILES: [CSK-DDI repository](https://github.com/cgwcbz2/CSK-DDI), based on the Deng et al. [DDIMDL benchmark](https://doi.org/10.1093/bioinformatics/btaa501), derived from DrugBank 5.1.3. The source files contain 37,264 pairs and 65 event IDs.
- Drug-target annotations: [DSimSynergy repository](https://github.com/hanjunwei-lab/DsimSynergy), `drug_target_9606.csv`. Target data and DDI labels may share DrugBank-derived information; temporal independence is not established.
- This repository does not redistribute the raw source data, pair-level training records, or predictions. `src/download_data.py` downloads the required source files and checks their SHA-256 hashes. Consult upstream terms before using or redistributing the data.
- Event descriptions were checked separately against the original [DDIMDL event database](https://github.com/YifanDengWHU/DDIMDL/blob/master/event.db). The mapping and audit summary are in `results_full/eight_class_mapping_verified.csv` and `results_full/eight_class_mapping_audit.json`. The source benchmark itself provides numeric labels.

## Experimental design

The sample includes the eight most frequent event classes, capped at 1,200 records per class, giving 9,320 pairs and 551 drugs. Per drug, features comprise a 512-bit radius-2 Morgan fingerprint and 1,519 binary target indicators; 536 drugs have at least one mapped target. Missing target annotations are encoded as zero, which does **not** prove absence of targets. Original drug-column order is retained.

Three input configurations (structure, target and combined) are evaluated across three seeds (11, 22, 33) and three partition scenarios: random pairs; S1, one drug held out and one observed in training; and S2, both drugs held out. All three algorithms use the same partitions for a given seed. The random and cold-start conditions also differ in training size and test composition. The splits overlap and are not independent external cohorts.

For combined features, macro-F1 (mean ± sample SD across the three overlapping splits):

| Model | Random pairs | S1: one unseen drug | S2: two unseen drugs |
|---|---:|---:|---:|
| Linear SVM | 0.821 ± 0.006 | 0.494 ± 0.046 | 0.177 ± 0.024 |
| Random forest | 0.798 ± 0.010 | 0.564 ± 0.034 | 0.218 ± 0.038 |
| XGBoost | 0.808 ± 0.008 | 0.533 ± 0.027 | 0.208 ± 0.032 |

These are descriptive results from this benchmark, not proof of generalizable model superiority. See `results_full/summary_metrics.csv`, `extended_metrics_summary.csv`, `paired_node_resampling.csv`, and `matched_mcnemar.csv` for the complete summaries.

## Run the study

Python 3.12 is recommended. The original package versions are recorded in `environment.txt`. The commands below assume the repository root is the working directory. On Windows, RDKit can alternatively be installed using conda-forge if `pip` installation is unavailable.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/download_data.py
python src/run_experiment.py
python src/analyze.py
python src/verify_results.py
python src/audit_structure_overlap.py
python src/rerun_scores.py
python src/statistical_comparison.py
```

`run_experiment.py` trains 81 model/representation/scenario/seed combinations and may take substantial time. The rerun step refits 27 combined-feature models, checks predicted classes against the original runs, and saves class scores for statistical analysis. Results may vary with software versions and computational environment; the exact source hashes are enforced by the download script.

For the optional full-data demonstration models and ordered-pair inference:

```bash
python src/train_final.py
python src/predict.py --help
```

These full-data models are fitted on the complete sample and do not have an independent test score. `predict.py` returns an event ID conditional on an already known interaction.

## Statistical interpretation

The study reports macro-F1 and complementary metrics, including one-versus-rest macro-AUROC. Three overlapping splits provide descriptive means and sample SDs, not independent replicates or confidence intervals. Paired drug-node resampling (3,000 draws) is conditional on fitted models and is exploratory; S2 ranges are suppressed because 730 of 3,000 draws lack at least one class. Exact McNemar tests compare correctness on drug-disjoint matched subsets, with Holm adjustment over 27 tests. Independence is not established and these p-values test accuracy rather than macro-F1. See `src/statistical_comparison.py` for the analysis.

To reproduce the post-hoc event mapping audit, manually obtain `event.db` from the original [DDIMDL repository](https://github.com/YifanDengWHU/DDIMDL/blob/master/event.db), save it as `data/original_ddimdl_event.db`, and run:

```bash
python src/verify_event_mapping.py
```

The script verifies the expected database hash, matches all 37,264 unique unordered drug pairs and checks that each of the 65 IDs corresponds to one mechanism/action combination. It does not assert that the benchmark is direction-independent or that any predicted interaction is clinically validated.

## Report and contents

- `src/`: download, feature construction, training, validation, statistics and report-generation scripts.
- `results_full/`: aggregate metrics, class mapping audit and figures; pair-level data and fitted models are not committed.
- `environment.txt`: package versions used for the reported run.

After the experiment and statistical analysis, generate the independent-study English report on Windows:

```bash
python src/create_final_report.py
```

The PDF is written to `output/`. The script uses the downloaded PubMed metadata and the aggregate results in `results_full/`, and requires Windows Times New Roman font files. The course logo is optional and is not distributed with the repository. Source data, pair-level predictions and trained model files are intentionally excluded; download or regenerate them as described above.

Primary references: Deng et al., *Bioinformatics* (2020), [doi:10.1093/bioinformatics/btaa501](https://doi.org/10.1093/bioinformatics/btaa501); Wang et al., *Bioinformatics* (2026), [doi:10.1093/bioinformatics/btag800](https://doi.org/10.1093/bioinformatics/btag800); Shen et al., *Bioinformatics* (2025), [doi:10.1093/bioinformatics/btaf569](https://doi.org/10.1093/bioinformatics/btaf569).
