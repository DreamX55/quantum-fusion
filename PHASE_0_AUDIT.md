# Q-FUSION: PHASE 0 AUDIT

**Date:** 2026-10-09
**Role:** Senior Research Engineer & Technical Lead

## 1. Repository Structure
- **Current State:** The repository (`c:\Users\maheit\Jonath\Quantum Fusion`) is completely empty. No foundational structure, configuration, or `.gitignore` exists.

## 2. Dataset
- **Current State:** No dataset is present in the workspace.
- **Action Required:** We need to acquire a relevant brain-tumour MRI dataset. The standard for T1, T2, FLAIR, and T1ce with segmentation masks is the **BraTS (Brain Tumor Segmentation) dataset**. We will need to set up a pipeline to download or link to a local copy of this dataset.

## 3. Modalities
- **Current State:** Unverified (Dataset missing).
- **Action Required:** Once BraTS is acquired, we will formally verify the alignment and existence of the 4 standard modalities (T1, T2, FLAIR, T1ce).

## 4. Segmentation Availability
- **Current State:** Unverified (Dataset missing).
- **Action Required:** BraTS includes multi-class segmentation masks (necrotic core, peritumoral edema, enhancing tumor). We will need to verify mask integrity and decide whether to use a multi-class objective or a simplified binary mask (whole tumor) for Objective 2.

## 5. Preprocessing
- **Current State:** No preprocessing pipeline exists.
- **Action Required:** We need to implement standard MRI preprocessing (e.g., N4 bias field correction, z-score normalization, cropping to non-zero brain regions, and potentially downsampling to accommodate quantum resource constraints).

## 6. Baseline
- **Current State:** No classical baselines exist.

## 7. Python Environment
- **Python Version:** 3.12.10
- **PyTorch:** Version 2.14.0+cpu is installed globally.
- **GPU Availability:** **CUDA: False**. The current environment defaults to CPU processing.
- **Action Required:** 
  - A dedicated virtual environment must be established.
  - If a GPU is physically present, PyTorch must be reinstalled with the correct CUDA bindings to ensure computational feasibility.

## 8. Quantum Environment
- **Current State:** No quantum frameworks (e.g., `qiskit`, `pennylane`) are currently installed. Minimal quantum circuit execution cannot proceed.
- **Action Required:** Define and install the quantum stack (recommend PennyLane for its seamless integration with PyTorch autograd).

## 9. Reproducibility State
- **Current State:** Blank slate. No seeds, no configs, no checkpoints.
- **Action Required:** Phase 1 must establish a rigorous configuration system (e.g., YAML + Hydra or simple dataclasses) and a logging framework (e.g., Weights & Biases or TensorBoard).

## 10. Major Blockers
1. **Missing Data:** No data available to build or test the classical baseline.
2. **Missing Quantum Frameworks:** Cannot run the minimal quantum sanity test.
3. **No GPU Acceleration Detected:** Training 3D/2D Multimodal CNN baselines on CPU will be extremely slow, severely bottlenecking the experimental pipeline.

## 11. Proposed Phase 1 Requirements (Immediate Next Steps)
Before moving to the Phase 1 Classical Multimodal Baseline, we must implement a **Phase 0.5 (Infrastructure & Data):**
1. **Environment Setup:** Create a `requirements.txt` or `environment.yml` installing `torch` (with CUDA if available), `pennylane`, `monai` (for medical imaging), `nibabel`, and `numpy`.
2. **Sanity Test:** Write a tiny script to verify PennyLane + PyTorch autograd works (the "minimal quantum sanity test" requirement).
3. **Data Acquisition:** Secure a minimal subset of BraTS (e.g., 5-10 patients) to construct the data loaders and verify shapes/modalities.
4. **Project Scaffolding:** Create modular directories (`data/`, `models/`, `experiments/`, `utils/`).

---
### Phase 0 Acceptance Criteria Checklist:
- [x] Repository is understood (Empty).
- [ ] Dataset is identified (BraTS proposed, missing).
- [ ] MRI modalities are verified (Pending Data).
- [ ] Modalities are checked for alignment (Pending Data).
- [ ] Segmentation masks are verified (Pending Data).
- [ ] Preprocessing is understood (Pending Data).
- [ ] Current baseline is identified (None exists).
- [x] Python environment is documented.
- [x] GPU availability is documented.
- [x] Quantum framework is verified (Missing).
- [ ] Minimal quantum circuit executes (Blocked).
- [ ] Quantum gradients work (Blocked).
- [ ] Reproducibility mechanism exists (Pending setup).
- [x] Major blockers are documented.
- [x] PHASE_0_AUDIT.md exists.
- [x] Phase 1 requirements are defined.
