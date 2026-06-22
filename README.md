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


## Overall Architecture of SARIF

### 🔑 Key Insights of the Overall Architecture 🔑

## Experiment Results

### Segmentation results on CASIAv2 training scheme

### Ablation study
