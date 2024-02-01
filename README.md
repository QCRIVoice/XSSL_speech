# XSSL_speech

This repository contains the code for the paper titled "SPEECH REPRESENTATION ANALYSIS BASED ON INTER- AND INTRA-MODEL SIMILARITIES" submitted to ICASSP 2024 Workshop XAISA.

## Code Availability

The code associated with this paper will be shared upon paper acceptance in the ICASSP 2024 Workshop XAISA.

## Abstract

Self-supervised models have revolutionized speech processing, achieving new levels of performance in a wide variety of tasks with limited resources. However, the inner workings of these models are still opaque. In this paper, we aim to analyze the encoded contextual representation of these foundation models based on their inter- and intra-model similarity, independent of any external annotation and task-specific constraint.

We examine different SSL models, varying their training paradigm – Contrastive (Wav2Vec2.0) and Predictive models (HuBERT); and model sizes (base and large). We explore these models on different levels of localization/distributivity of information, including:
1. Individual neurons
2. Layer representation
3. Attention weights
4. Compare the representations with their fine-tuned counterparts.

Our results highlight that these models converge to similar representation subspaces but not to similar neuron-localized concepts.

## Additional Information

For further details and access to the code, please refer to the paper accepted in the ICASSP 2024 Workshop XAISA.

## Citation

If you find this work useful, please consider citing our paper.

