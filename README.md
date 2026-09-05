# Master_Thesis_EWI_2026

This is the repository for my Master-Thesis ```Multivariate Generation of Synthetic Electricity and Heatload Profiles for Residential Buildings Using Diffusion Modells```. 
The initial data generation of the synthetic trainingsdata has been done via a highly customized version of [TSIB](https://github.com/NV717/tsorb_simuflex). The exact data can be reproduced via the customized ````TSIB````-generator
and the [Generation-Manifest](DataPreparation/Generation/manifest.parquet).
Further structure and content of this repo will be added and updated as time goes on.

## Planned Thesis Structure
0. Validate Generation data
    - make sure there are no problems


1. Data partitioning
    - group aware, stratified over archetypes with same distributions for train/val/test


2. Data sampling
    - Methodology to create weekly windows (function with length parameter)


3. Deskriptive Analysis and evluations Framework
    - Analysis of the yearly data before windowing
    - Analysis of Windowed data and establishment of reference metrics
        - temporal, multivariat, conditional statistics
        - 1D (weekly) vs 2D (yearly) via weekly connections


4. Evaluation and Definition of the Condition Metrics
    - define which params to use for condition and in what why (dynamic/ static)
    - possible redundancys (temperature-ts <-> cliate Region)


5. evluations Framework
    - pipeline for later evaluation (Half-Split-bootstrap etc.)
    - Metrics: based on 3. TBD

evtl. change 5 and 6 if metric need normalized values

6. Normalization based on results from 3.
    - channelwise normalization
    - only training data


7. Benchmarks
    - Resampling etc. to create a benchmark for the validation harness and compare to the model later (random sampling etc)


8. Diffusion Modell
    - cascade:
        1. occupacy model
        2. unconditioned single channel load model
        3. unconditioned multi channel load model
        4. conditioned multi channel load model with "real" occ as ground truth evtl. conditioned multi channel load model without occ
        5. conditioned multi channel load model with generated occ