# SARIF [2026ECCV]
The reproduction code of SARIF which is accepted in ECCV 2026 !

## 🔔 Latest News
- [2026-06-22]: We uploaded SARIF Official Code!
- [2026-06-22]: We are happy to announce that SARIF was accepted at [ECCV 2026]!🥳🥳

## Poster
-- The poster is scheduled to be released soon!

## Abstract
Image forgery localization remains challenging due to diverse manipulation techniques and distribution shifts. Existing recent forgery localization models achieve high accuracy on benchmarks but often struggle with cross-domain generalization and robustness. In this paper, we propose \textbf{SARIF (Segment Anything for Robust Image Forensics)}, a framework that leverages Segment Anything Model (SAM), which has a promptable architecture and generalization ability to overcome these limitations. SARIF introduces a feedback-guided mask decoder and a dual-encoder design that extracts forgery-specific information to capture forensic traces while exploiting SAM’s architecture. To localize manipulated regions, we design a block-wise prompting mechanism that derives forgery-specific cues from residual features between an adapted encoder and its frozen counterpart. These features are fused with the previous mask prompt to drive a feedback-based mask refinement process, enabling automatic forgery segmentation without manual input. Extensive experiments on standard forgery-localization benchmarks show that SARIF achieves strong average cross-dataset performance and robustness to common image corruptions.

## 🔑 Key Motivation 🔑
<img width="5002" alt="Motivation" src="https://github.com/user-attachments/assets/5234e648-8a34-400d-9a6f-71213413cb26" />

Without the adapter, the encoder focuses on semantic content and misses subtle manipulation artifacts, causing imprecise masks. With the adapter, it learns forgery-specific cues that guide the decoder to produce sharper and more accurate localization. Concretely, this domain gap between the adapted and the frozen encoder represents the adapter-learned forgery-specific information.

## Overall Architecture of SARIF

### 🔑 Key Insights of the Overall Architecture 🔑

## Experiment Results

### Segmentation results on CASIAv2 training scheme

### Ablation study
