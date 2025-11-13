Deep learning based biliary stent classification and transfer learning adaptation to an additional stent type
============================================================================================================

Overview
--------
This repository contains the training, fine-tuning, and visualization code for a deep learning system that classifies biliary stents from endoscopic images and adapts (via transfer learning) to an additional stent type. The codebase supports:
- Supervised classification with timm models (e.g., EfficientNet, ResNet, ViT/DeiT)
- MLflow experiment tracking and model artifact logging
- Automatic Grad-CAM export on the validation set
- ROC/AUC analysis across epochs and sets, DeLong tests, and confusion matrices

You will add main figures and Grad-CAM results as assets when available; placeholders are referenced below.

Repository layout
-----------------
- `training/`
  - `finetune.py`: main fine-tuning and evaluation pipeline with MLflow logging and Grad-CAM export
  - `engine_finetune.py`: one-epoch training, evaluation, and metrics computations
  - `util/*`: helpers (dataset building, schedulers, metrics) referenced by the training scripts
- `visualization/`
  - `roc_ep.py`, `roc_ep_single.py`: ROC/AUC across epochs and sets (multi- and single-case variants)
  - `delong_test_adaptive_heatmap.py`: DeLong tests and meta-analysis across epochs; adaptive heatmaps
  - `single_con.py`: confusion matrix visualization using Grad-CAM CSV outputs

Figures (placeholders you will replace)
---------------------------------------
Place your paper-ready figures under:
- `assets/figures/main_figure.png`  (Main Figure)
- `assets/figures/gradcam_example.png`  (Representative Grad-CAM)
- Optionally:
  - `assets/figures/roc_curves_case1.png`, `assets/figures/roc_curves_single.png`
  - `assets/figures/confusion_matrix_average.png`
  - `assets/figures/all_cases_adaptive_heatmap.png`

Example embeds (will render once files are added):
![Main Figure](assets/figures/main_figure.png)
![Grad-CAM](assets/figures/gradcam_example.png)

Environment
-----------
Recommended versions (others may work):
- Python 3.9+
- PyTorch (CUDA if available)
- torchvision
- timm
- mlflow
- scikit-learn, matplotlib, seaborn
- pandas, numpy, tqdm

Quick setup:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA
pip install timm mlflow scikit-learn matplotlib seaborn pandas numpy tqdm pillow
```

Data structure
--------------
The dataset is expected as ImageFolder-style with train/val splits:
```
<DATA_PATH>/
  train/
    <class_0>/*.jpg|png
    <class_1>/*.jpg|png
    ...
  val/
    <class_0>/*.jpg|png
    <class_1>/*.jpg|png
    ...
```
Set `--nb_classes` to match the number of folders (classes).

Training and fine-tuning
------------------------
`training/finetune.py` uses timm to create a model, fine-tunes on your dataset, logs metrics to MLflow, and (optionally) exports Grad-CAMs for the validation set.

Common arguments (subset):
- `--model_name` (e.g., `efficientnet_b1.ft_in1k`, `resnet50.a1_in1k`, `deit_base_patch16_224`)
- `--data_path` (root folder containing `train/` and `val/`)
- `--nb_classes` (number of classes)
- `--epochs`, `--batch_size`, `--blr` (base LR), `--weight_decay`, `--smoothing`
- `--mlflow_server`, `--mlflow_experiment`, `--run_name`, `--mlflow_account_name`, `--mlflow_account_password`
- `--device` (`cuda` or `cpu`)
- Optional: `--ckpt_path` (load a saved model and rewire the classifier head to new classes)

Example (EfficientNet, 5 classes):
```bash
python training/finetune.py ^
  --model_name efficientnet_b1.ft_in1k ^
  --data_path D:\your_dataset ^
  --nb_classes 5 ^
  --epochs 300 ^
  --batch_size 64 ^
  --blr 1e-3 ^
  --mlflow_server http://YOUR_MLFLOW:5000 ^
  --mlflow_experiment "Biliary Stent" ^
  --run_name "effb1-5cls"
```

Notes:
- If you provide `--ckpt_path`, the script loads the model and replaces the classifier head for EfficientNet, MobileNet, ResNet, or ViT/DeiT automatically.
- MLflow username/password can be provided via environment or arguments.
- The script computes accuracy, precision, recall, and F1 (macro) at each epoch.

Grad-CAM export
---------------
At the end of training, `finetune.py` automatically:
1) Selects a suitable target layer per model family (EfficientNet, MobileNet, ResNet, DeiT/ViT)
2) Runs Grad-CAM on the validation set
3) Saves images and a CSV summarizing `[file, original_label, predicted_class, confidence]`
4) Archives and logs the results to MLflow

Configure output naming via:
- `--zip_file_name` (base name for the Grad-CAM output; the script will create a results folder and a ZIP artifact)

After you generate representative results, place 1–2 key Grad-CAM panels under `assets/figures/` and reference them in the paper and README.

Visualization and statistics
----------------------------
The `visualization/` folder provides post-hoc analysis utilities. These scripts currently contain path variables for datasets and checkpoints—adjust them to your environment before running.

1) ROC and AUC across epochs and sets
- `visualization/roc_ep.py` (multi-case)
- `visualization/roc_ep_single.py` (single-case)

What they do:
- Load per-epoch checkpoints for each set and compute per-set ROC curves and AUC
- Average across sets to yield epoch-wise summary AUCs and curves
- Save epoch AUC bar plots and ROC overlays

Edit at the top of each script:
- Dataset roots (e.g., `case_root_paths` or `case_root_path`)
- Weight roots (e.g., `weight_root_paths` or `weight_root_path`)
- `model_name`, `num_classes`, and `epochs`

2) DeLong tests and adaptive heatmaps
- `visualization/delong_test_adaptive_heatmap.py`

What it does:
- For each case and epoch, runs inference, computes ROC/AUC
- Performs DeLong tests between epoch pairs within each set
- Aggregates across sets (meta-analysis) and saves adaptive p-value heatmaps

3) Confusion matrices from Grad-CAM CSVs
- `visualization/single_con.py`
  - Reads per-set Grad-CAM CSVs (e.g., `*_gradcam_results.csv`), builds confusion matrices, and saves PNGs
  - Edit `base_path` to the directory containing your CSVs and update `CLASSES` to match your label names

Transfer learning to an additional stent type
---------------------------------------------
To adapt a model to a new stent type (adding a class or retargeting an existing label set):
1) Prepare data folders (train/val) with the new class taxonomy.
2) Set `--nb_classes` to the updated count.
3) Optionally start from a previous checkpoint via `--ckpt_path` (the classifier head will be re-wired automatically).
4) Re-run `finetune.py` and re-generate Grad-CAMs and analysis figures.

Reproducibility tips
--------------------
- Fix random seeds via `--seed` (the script also adjusts per-rank).
- Use consistent image preprocessing across training and evaluation.
- Log all hyperparameters to MLflow and export artifacts (models, Grad-CAM ZIP).
- When comparing epochs, keep dataset splits and evaluation code identical.

Results placeholders (to be replaced with your final figures)
-------------------------------------------------------------
- Main performance figure: `assets/figures/main_figure.png`
- Representative Grad-CAM: `assets/figures/gradcam_example.png`
- Average confusion matrix: `assets/figures/confusion_matrix_average.png`
- Adaptive heatmap of DeLong tests: `assets/figures/all_cases_adaptive_heatmap.png`

Citation
--------
If you use this repository, please cite:

Deep learning based biliary stent classification and transfer learning adaptation to an additional stent type  
Authors: [Add authors here]  
Year: [Year]  
Venue: [Journal/Conference]  

License
-------
This project is licensed under the terms of the license in `LICENSE`.

Contact
-------
For questions or issues, please open an issue or contact the maintainers.


