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
- **Collaborator contribution:** the ResNet-18 model is assigned to another Group 38 member. `src/models/resnet18.py`
  is currently a reference implementation (no collaborator code was in the repository on 2026-11-25). Update this
  section when their implementation/checkpoint is integrated.

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
Claude Code (Anthropic, model Claude Opus 5.5) was used on 2026-11-25 to implement the repository scaffolding, data
pipeline, training/evaluation infrastructure, tests and documentation, following the group's written specification
(`ENSF 519.docx`) and approved proposal. The group must review, understand and take responsibility for this code and
state the AI assistance in the report. It must not be presented as entirely unaided student work.
