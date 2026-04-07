# ResMatching

Official implementation of **ResMatching**, accepted at [ISBI 2026](https://biomedicalimaging.org/2026/).

> **ResMatching: Noise-Resilient Computational Super-Resolution via Guided Conditional Flow Matching**  
> Anirban Ray, Vera Galinova, Florian Jug
> [[arXiv]](https://arxiv.org/abs/2510.26601)
> 
![Figure 1](assets/figure1.png)

## Abstract

Computational Super-Resolution (CSR) in fluorescence microscopy has, despite being an ill-posed problem, a long history. At its very core, CSR is about finding a prior that can be used to extrapolate frequencies in a micrograph that have never been imaged by the image-generating microscope. It stands to reason that, with the advent of better data-driven machine learning techniques, stronger prior can be learned and hence CSR can lead to better results. Here, we present **ResMatching**, a novel CSR method that uses guided conditional flow matching to learn such improved data-priors. We evaluate ResMatching on 4 diverse biological structures from the BioSR dataset and compare its results against 7 baselines. ResMatching consistently achieves competitive results, demonstrating in all cases the best trade-off between data fidelity and perceptual realism. We observe that CSR using ResMatching is particularly effective in cases where a strong prior is hard to learn, e.g. when the given low-resolution images contain a lot of noise. Additionally, we show that ResMatching can be used to sample from an implicitly learned posterior distribution and that this distribution is calibrated for all tested use-cases, enabling our method to deliver a pixel-wise data-uncertainty term that can guide future users to reject uncertain predictions.

## Installation

```bash
pip install uv
uv sync
```

## License

MIT

