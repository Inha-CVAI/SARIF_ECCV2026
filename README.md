> Dong-Hyun Moon<sup>&#42;</sup>, Ju-Hyeon Nam<sup>&#42;</sup>, Sang-Chul Lee<sup>&dagger;</sup>  
> [![Github](https://img.shields.io/badge/-Github-black?style=flat&logo=github&logoColor=white)](https://github.com/Inha-CVAI/SARIF_ECCV2026)
> [![Stars](https://img.shields.io/github/stars/Inha-CVAI/SARIF_ECCV2026?style=social)](https://github.com/Inha-CVAI/SARIF_ECCV2026/stargazers)
> [![arXiv](https://img.shields.io/badge/arXiv-2606.21108-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2606.21108)  
> <sup>&#42;</sup> Equal contribution.  
> <sup>&dagger;</sup> Corresponding author.

# SARIF [2026 ECCV]
The reproduction code of SARIF which is accepted in ECCV 2026 

## 🔔 Latest News
- [2026-06-22]: We uploaded SARIF Official Code!
- [2026-06-22]: We are happy to announce that SARIF was accepted at [ECCV 2026]!🥳🥳

## Poster

-- The poster is scheduled to be released soon!

## Abstract
Image forgery localization remains challenging due to diverse manipulation techniques and distribution shifts. Existing recent forgery localization models achieve high accuracy on benchmarks but often struggle with cross-domain generalization and robustness. In this paper, we propose **SARIF (Segment Anything for Robust Image Forensics)**, a framework that leverages Segment Anything Model (SAM), which has a promptable architecture and generalization ability to overcome these limitations. SARIF introduces a feedback-guided mask decoder and a dual-encoder design that extracts forgery-specific information to capture forensic traces while exploiting SAM’s architecture. To localize manipulated regions, we design a block-wise prompting mechanism that derives forgery-specific cues from residual features between an adapted encoder and its frozen counterpart. These features are fused with the previous mask prompt to drive a feedback-based mask refinement process, enabling automatic forgery segmentation without manual input. Extensive experiments on standard forgery-localization benchmarks show that SARIF achieves strong average cross-dataset performance and robustness to common image corruptions.

## 🔑 Key Motivation 🔑
<p align="center">
  <img width="40%" alt="Motivation" src="https://github.com/user-attachments/assets/5234e648-8a34-400d-9a6f-71213413cb26" />
</p>

Without the adapter, the encoder focuses on semantic content and misses subtle manipulation artifacts, causing imprecise masks. With the adapter, it learns forgery-specific cues that guide the decoder to produce sharper and more accurate localization. Concretely, this domain gap between the adapted and the frozen encoder represents the adapter-learned forgery-specific information.

_**SARIF exploits the discrepancy between the frozen SAM representation and the LoRA-adapted representation as a forgery-specific prompt, allowing the decoder to focus on manipulation-aware evidence rather than generic semantic cues.**_

## Overall Architecture of SARIF

<p align="center">
  <img width="66%" alt="model" src="https://github.com/user-attachments/assets/73adf6cd-53ad-4343-88e3-1af7948030c6" />
</p>

The overall architecture of the proposed SARIF. (a) Fine-tuned SAM Image Encoder. (b) Original SAM Image Encoder. (c) Feedback-Guided Mask Decoder. (d) Forgery-Specific Information Extractor. (e) Notation description used in this paper.

## 🔑 Key Insights of the Overall Architecture 🔑
- _**Dual-Encoder Residuals for Extracting Forgery-Specific Cues**_:
SARIF employs two SAM encoders: a frozen SAM encoder and a fine-tuned SAM encoder. The frozen encoder preserves SAM’s broad semantic and structural knowledge, while the LoRA-adapted encoder shifts toward manipulation-sensitive patterns. By comparing these two representations, SARIF extracts forgery-specific information from the 5th, 11th, 17th, and 23rd SAM global-attention blocks and the final image embedding. This information is fused with the previous mask prompt to form a task-specific prompt, enabling the mask decoder to refine the predicted mask and produce sharper forgery localization.

- _**Feedback-Guided Prompt-Mask Refinement**_:
SARIF fully exploits SAM’s promptable decoder by converting each previous mask prediction into a mask prompt and fusing it with forgery-specific information. This prompt-mask feedback loop progressively refines the segmentation result, sharpening boundaries, suppressing spurious regions, and accumulating forensic evidence across refinement stages without requiring manual points, boxes, or masks.

- _**Robust Foundation-Model Adaptation for Practical Image Forensic**_:
By combining parameter-efficient SAM adaptation, hierarchical residual cue extraction with feedback-guided decoding, SARIF improves cross-domain localization and maintains competitive robustness under various distortion settings and modern forgery patterns.

## Experiment Results

### Segmentation results on CASIAv2 training scheme
<table>
  <tr>
    <td width="50%" align="center">
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/0b1f924e-7b26-4a8f-a54c-a4ce4b1abf12" />
    </td>
    <td width="50%" align="center">
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/d55ca124-2c6f-4527-9131-1d658cc8726d" />
    </td>
  </tr>
</table>


### Ablation study
