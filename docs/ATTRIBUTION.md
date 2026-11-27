# Attribution

## Original project work (Group 38)
- Research question, experimental design, custom four-block CNN architecture (`src/models/cnn.py`), shared
  data/training/evaluation pipeline and analysis.

## Reused architectures, weights and data
- **ResNet-18:** K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016
  (arXiv:1512.03385). Implementation: Torchvision `torchvision.models.resnet18`.
- **Pretrained weights:** Torchvision `ResNet18_Weights.IMAGENET1K_V1`, trained on ImageNet-1k
  (O. Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge," IJCV 2015).
- **GenImage dataset:** M. Zhu, H. Chen, Q. Yan, X. Huang, G. Lin, W. Li, Z. Tu, H. Hu, J. Hu, Y. Wang, "GenImage: A
  Million-Scale Benchmark for Detecting AI-Generated Image," NeurIPS 2023 (arXiv:2306.08571);
  https://github.com/GenImage-Dataset/GenImage, CC BY-NC-SA 4.0. Its real images come from ImageNet.
- **Collaborator contribution (Group 38 ResNet-18 project, "Spotting the Unseen Generator").** Their project is kept
  unmodified under `collaborator/resnet18_project/`: code, notebook, tests, three-seed results and figures. What was
  integrated from it:
  - **Model construction:** `src/models/resnet18.py` reproduces their `build_resnet18`: Torchvision ResNet-18,
    `IMAGENET1K_V1`, a new `Linear(512, 2)` head with default initialisation, full fine-tuning, 0 = real / 1 = AI.
  - **Dataset source:** the choice of the Hugging Face mirror `nebula/GenImage-arrow`, the pinned revision, and the
    fetch/retry and Arrow-row reading code, adapted in `src/data/genimage_hf.py`. Their "first N rows" selection was
    replaced by seeded hash sampling (see DATASET.md).
  - **Their own results:** reported separately as their recipe. `results/tables/collaborator_resnet18_runs.csv` gives
    mean ± std over their seeds 0–1–2. Those numbers come from their own splits and preprocessing, so they are not
    directly comparable with E2.

  E2 trains their model definition under this repository's shared protocol (same manifests, preprocessing, optimiser
  settings and early stopping as the CNN), so the CNN-vs-ResNet comparison is controlled.
- **Dataset mirror:** `nebula/GenImage-arrow` on Hugging Face, an Arrow re-packaging of GenImage. Its licence follows
  GenImage (CC BY-NC-SA 4.0).

## Libraries
PyTorch (A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019),
Torchvision, NumPy, Pandas, scikit-learn (metrics, logistic regression), SciPy (exact binomial test), Matplotlib,
Pillow, ImageHash (perceptual hashing), PyYAML, pytest.

## Related work cited in the proposal
- S.-Y. Wang, O. Wang, R. Zhang, A. Owens, A. A. Efros, "CNN-Generated Images Are Surprisingly Easy to Spot... for
  Now," CVPR 2020 (arXiv:1912.11035).
- P. Grommelt, L. Weiss, F.-J. Pfreundt, J. Keuper, "Fake or JPEG? Revealing Common Biases in Generated Image
  Detection Datasets," ECCV Workshops 2024 (arXiv:2403.17608).

## AI assistance
Claude Code (Anthropic, model Claude Opus 5.5) was used on 2026-11-25 and 2026-11-26 to implement the repository scaffolding, data
pipeline, training/evaluation infrastructure, tests and documentation, following the group's written specification
(`ENSF 519.docx`) and approved proposal. The group must review, understand and take responsibility for this code and
state the AI assistance in the report. It must not be presented as entirely unaided student work.
