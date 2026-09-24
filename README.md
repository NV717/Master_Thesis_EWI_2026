# Master_Thesis_EWI_2026

This is the repository for my Master-Thesis ```Multivariate Generation of Synthetic Electricity and Heatload Profiles for Residential Buildings Using Diffusion Modells```. 
The initial data generation of the synthetic trainingsdata has been done via a highly customized version of [TSIB](https://github.com/NV717/tsorb_simuflex). The exact data can be reproduced via the customized ````TSIB````-generator
and the [Generation-Manifest](DataPreparation/Generation/manifest.parquet).
Further structure and content of this repo will be added and updated as time goes on.

## Planned Thesis Structure

1. Data Preparation
    - Data Validation
    - group aware, stratified over archetypes with same distributions for train/val/test
    - Data Windowing (1 day, 1 week etc.)

2. Deskriptive Analysis and evluations Framework
    - Analysis of the yearly data before windowing
    - Analysis of Windowed data and establishment of reference metrics
        - temporal, multivariat, conditional statistics
        - 1D (weekly) vs 2D (yearly) via weekly connections

3. Evaluations Framework
    - pipeline for later evaluation (Half-Split-bootstrap etc.)
    - Metrics: based on 2. TBD
    - define which params to use for condition and in what why (dynamic/ static)
    - possible redundancys (temperature-ts <-> cliate Region)
    - Split Half Bootstrap as baseline

4. Diffusion Modell
    - cascade:
        1. occupacy model
        2. unconditioned single channel load model
        3. unconditioned multi channel load model
        4. conditioned multi channel load model with "real" occ as ground truth evtl. conditioned multi channel load model without occ
        5. conditioned multi channel load model with generated occ
    - Maybe 2D Version as Ablation
    - Maybe use model architecture for real data

5. Benchmarks
    - serves benchmark for the validation harness and compare to the model later
    - random sampling
    - T-Copula


