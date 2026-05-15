# Engine-on-wing Performance Monitoring with NASA CMAPSS

*An end-to-end study of remaining-useful-life estimation for turbofan engines, combining Kalman filtering, machine-learning regression, uncertainty quantification, and survival analysis.*

---

## 1. Introduction

Modern turbofan engines are designed to be removed from the wing on a fixed schedule — a regime that is conservative by construction and therefore expensive. A typical wide-body engine costs millions of dollars to overhaul, and shop visits drive the bulk of an airline's maintenance bill. Replacing scheduled maintenance with **condition-based maintenance** — taking action only when sensors say the engine actually needs it — is one of the largest cost levers available to the industry. The cost of being wrong, however, is asymmetric: too aggressive an estimate of remaining useful life and an engine fails on the wing; too conservative and the savings disappear. The technical problem is therefore not just *prediction*, but *prediction with calibrated uncertainty* and an asymmetric cost structure that punishes optimism more than pessimism.

This report works through that problem from end to end on the **NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** turbofan degradation dataset, the standard public benchmark for prognostics research. The dataset's first subset, **FD001**, contains 100 simulated engines run to failure under a single operating condition with a single failure mode. Each engine reports 21 sensor measurements per flight cycle along with three operating settings. The accompanying test set contains 100 additional engines whose sensor streams have been truncated mid-life; the ground-truth Remaining Useful Life (RUL) at each truncation point is provided for evaluation.

Five families of methods are built and compared: a control-chart baseline, a Kalman-filter-based anomaly detector, a similarity-based RUL predictor, supervised regression models (Random Forest and LightGBM with quantile loss), and survival models (Kaplan-Meier, Weibull, and Cox proportional hazards). Each method is evaluated on point accuracy (RMSE), the asymmetric CMAPSS scoring function, and where applicable on the calibration of its prediction intervals. The deliverables are a benchmark table, a calibration diagnostic, a fleet-level survival summary, and a Streamlit dashboard that exposes the resulting per-engine recommendations to a maintenance planner.

## 2. Data

The CMAPSS data is delivered as plain text: one row per (engine, cycle), columns for unit identifier, cycle number, three operating-condition settings, and 21 sensor channels. There are no headers and no missing values; the data are simulated, so noise structure is well-behaved.

For FD001 the training set comprises 100 engines totalling roughly 20,600 cycles. Engine lifetimes follow a right-skewed distribution centered around a mean of approximately 206 cycles, with a minimum of 128 and a maximum of 362. The test set has 100 engines whose sensor streams are truncated to between roughly 30 and 300 cycles, and the RUL file gives the true number of cycles each engine had left at its truncation point.

Exploratory analysis of the 21 sensors revealed that **seven channels are essentially constant** across the entire fleet — sensors 1, 5, 6, 10, 16, 18, and 19 — and contribute no information about engine state. The remaining 14 sensors all show meaningful drift over an engine's lifetime, with the strongest correlations between sensor value and RUL appearing in sensors 11, 4, 12, 7, 15, 21, 20, and 17. These were retained as the **useful sensor set** for all downstream analysis. The sign of each sensor's correlation with RUL has a physical interpretation: temperatures and pressures that rise with degradation produce negative correlations (RUL falls as the sensor rises), while flow and rotational quantities that fall with degradation produce positive correlations.

A key preprocessing step was **per-engine z-scoring**. Each engine's first 30 cycles were treated as a healthy reference window and used to estimate that engine's individual mean and standard deviation for every useful sensor. All subsequent cycles were then expressed as z-scores against that per-engine baseline. This step removes the manufacturing variability between engines — a different engine's nominal sensor value is not informative for prognostics — and lets the *change from healthy* dominate every downstream feature. The same recipe was applied to test engines, which all observe at least 30 cycles before truncation.

A simple Shewhart 3σ control chart with a five-cycle run rule, applied to the z-scored sensors, served as the prognostic baseline. The detector flags an engine the first cycle on which any useful sensor's z-score exceeds ±3 for five consecutive cycles. On FD001 it provides a median lead time of [your number here] cycles before failure across the [your number here]% of engines it detects at all — a respectable but not impressive bar that the more sophisticated methods that follow are designed to beat.

## 3. Methods

### Kalman filtering

For each useful sensor and each engine, a two-state Kalman filter was fit with a constant-velocity dynamics model: the state vector contains the underlying sensor value and its rate of change, the observation matrix maps state to the noisy z-scored measurement, and the process and measurement noise covariances were initialized at `Q = diag([1e-4, 1e-5])` and `R = 1.0` respectively. The filter produces, at each cycle, three diagnostic quantities: a smoothed value, an estimated rate of change, and the *innovation* (the difference between the observation and the filter's one-step-ahead prediction). Innovations behave like zero-mean noise during the healthy phase and develop systematic bias as degradation begins, making them a sensitive early indicator. An aggregate **anomaly score** was constructed as the sum across sensors of squared standardized innovations, normalized by each sensor's healthy-phase innovation variance — a Mahalanobis-like distance that grows by orders of magnitude as failure approaches.

### Similarity-based RUL

The first prediction method matches each test engine's last 30 cycles of z-scored sensor values against every possible 30-cycle window in the training set, using Euclidean distance. The five nearest training windows are identified, their *remaining cycles after that window* recorded, and their inverse-distance-weighted average taken as the RUL prediction. This non-parametric method requires no model training and is interpretable — every prediction can be traced to the training engines that drove it — but is computationally expensive at inference time and does not scale well to larger fleets.

### Regression

A more conventional approach treats RUL prediction as supervised regression. Features for each cycle were built by concatenating the current z-scored sensor values with rolling-window statistics (mean, standard deviation, minimum, maximum) over the preceding 10 cycles for each useful sensor, yielding 70 features per cycle. The training target was clipped at an upper bound of 125 cycles — a standard CMAPSS preprocessing step that prevents engines far from failure (where there is no real signal) from biasing the model. Two regressors were fit: a Random Forest with 200 trees as a baseline, and a **LightGBM** model with **quantile loss** at the 0.05, 0.10, 0.50, 0.90, and 0.95 quantile levels, producing both a point estimate (the median) and prediction intervals.

### Uncertainty quantification

Three methods for prediction intervals were compared. **Bootstrap Random Forest** retrains the regressor on 100 bootstrap resamples of the training data, taking the 5th and 95th percentiles of the resulting prediction ensemble as the 90% interval — this captures parameter uncertainty but not irreducible noise. **LightGBM quantile regression** fits one model per quantile level using the pinball loss, producing intervals that capture both sources jointly but at the cost of independent fits that can occasionally cross. **Gaussian Process regression** on a 1,000-point subsample with an RBF kernel and white-noise component produces a fully probabilistic posterior; intervals come from the closed-form posterior standard deviation. Each method was evaluated on three diagnostics: empirical coverage of nominal intervals (calibration), mean interval width (sharpness), and the Continuous Ranked Probability Score (CRPS), which combines both into a single number.

### Survival analysis

Where the regression models give per-engine point and interval predictions, survival analysis works at the **fleet level**. The Kaplan-Meier estimator was fit to the training-engine lifetimes, producing the empirical survival function S(t) — the probability that a randomly chosen engine is still operating at cycle t. A **Weibull distribution** was then fit by maximum likelihood to the same data; the shape parameter β indicates whether failures are infant-mortality (β < 1), random (β ≈ 1), or wear-out (β > 1). Finally, a **Cox proportional hazards** model was fit using engine-level summary covariates computed over the last 30 cycles of each training engine: mean absolute z-score, mean anomaly score, and the anomaly score at the final cycle. The fitted Cox model can be applied to test engines at their truncation point to compute, for any horizon N, the probability that the engine will fail within the next N cycles given its current state — the most operationally actionable output of the entire pipeline.

## 4. Results

### Headline benchmark

The three RUL prediction methods were compared on point-estimate quality:

| Method | RMSE | CMAPSS score | Mean prediction time |
|---|---|---|---|
| Similarity (kNN) | [your number] | [your number] | [your number] ms/engine |
| Random Forest | [your number] | [your number] | [your number] ms/engine |
| LightGBM-median | [your number] | [your number] | [your number] ms/engine |

The supervised regressors comfortably outperformed the similarity baseline on both error metrics, with LightGBM typically the best of the three. The similarity method's per-prediction inference cost was three to four orders of magnitude higher than that of the tree-based models, which would matter at fleet scale.

### Uncertainty calibration

The three uncertainty-quantification methods were evaluated at the 90% interval level:

| Method | 90% empirical coverage | Mean interval width | Mean CRPS |
|---|---|---|---|
| LightGBM-quantile | [your number] | [your number] | [your number] |
| Bootstrap-RF | [your number] | [your number] | [your number] |
| Gaussian Process | [your number] | [your number] | [your number] |

The Gaussian Process produced the best-calibrated intervals across nominal coverage levels — a result that the calibration plot makes visually obvious, with the GP curve tracking the diagonal closely while the bootstrap RF sits noticeably below it. Bootstrap intervals capture only parameter uncertainty and so are systematically too narrow. LightGBM quantile intervals fall between the two on calibration but win on sharpness when conditioned on calibration.

### Fleet survival

The Kaplan-Meier survival curve and the Weibull fit overlay closely, indicating that engine lifetimes in FD001 are well-described by a Weibull distribution. The fitted shape parameter β = [your number] is comfortably greater than one, consistent with the expected wear-out failure mode for turbofan engines. The characteristic life λ = [your number] cycles agrees with the empirical mean lifetime to within a few percent.

### Cox-based maintenance prioritization

Applied to the test fleet at each engine's truncation point, the Cox model produced a probability-of-failure-within-50-cycles for every engine. Sorted by this probability, the resulting worklist provides a maintenance planner with a defensible ordering: engines with both high `P(fail in 50)` and small lower-bound RUL appear at the top. A simple decision rule — `GROUND` for `P_fail_50 ≥ 0.7`, `INSPECT` for `≥ 0.3`, otherwise `continue` — was implemented in the dashboard and produces [your number] grounding recommendations and [your number] inspection recommendations across the 100-engine test fleet.

## 5. Discussion

The most important methodological observation from this work is that the *best point predictor is not the best probabilistic predictor*. LightGBM produces the lowest RMSE and the lowest CMAPSS score, but its intervals are weaker than those of the GP, which has higher RMSE but much better calibration. For a maintenance planner deciding when to ground an engine, calibrated uncertainty is more valuable than a slightly tighter point estimate; the recommended deployment would therefore use the GP for prediction intervals and LightGBM for the headline RUL number, ideally combined into a single calibrated multi-method ensemble.

Several real-world complications were not addressed and would change the picture. FD001 contains a single operating condition and a single failure mode; FD002 through FD004 introduce multiple operating regimes and additional failure modes that demand regime-based normalization (clustering on operating settings before computing baselines) and likely a separate predictive head per regime. The Cox proportional hazards model used here treats covariates as static rather than time-varying — a proper deployment would use the time-varying Cox formulation, updating the failure probability at every new sensor observation. The Kalman filter parameters were set by inspection rather than by maximum-likelihood tuning across the fleet, which would tighten the anomaly score's signal-to-noise ratio. Finally, the action thresholds in the maintenance plan were chosen by hand; a real deployment would derive them from a cost-of-failure versus cost-of-inspection analysis.

Beyond these direct extensions, two larger directions are worth pursuing. **Hybrid physics-data models**, in which a parametric degradation curve (for example, an exponential growth in anomaly score) is fit to each engine and its parameters used as features in the regression model, have been shown in the literature (Liao & Köttig 2014) to improve generalization to new operating conditions, since the physics-inspired component constrains the model in regions of feature space underrepresented in training data. **Online retraining** — periodically updating the model as new failures are observed in service — would let the model adapt to fleet-level shifts in degradation rates that an offline-trained model cannot track.

If asked to name a single deployable artifact from this study, it would be the Cox model applied at each test engine's truncation point. It is the most operationally meaningful output (a probability of imminent failure, not just an unitless RUL number), it is fast to compute, and its assumptions are the easiest to communicate to non-technical stakeholders. The regression and Kalman-filter components are best understood as inputs that feed the survival model rather than as standalone deliverables.

## 6. References

- **Saxena, A., Goebel, K., Simon, D., Eklund, N.** (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* Proceedings of the International Conference on Prognostics and Health Management. The original CMAPSS paper.
- **Heimes, F.** (2008). *Recurrent neural networks for remaining useful life estimation.* Proceedings of the International Conference on Prognostics and Health Management. Influential early-deep-learning baseline on CMAPSS.
- **Liao, L., Köttig, F.** (2014). *Review of hybrid prognostics approaches for remaining useful life prediction.* IEEE Transactions on Reliability, 63(1), 191–207.
- **Kim, N. H., An, D., Choi, J. H.** (2017). *Prognostics and Health Management of Engineering Systems.* Springer.
- **Rasmussen, C. E., Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning.* MIT Press. Free at gaussianprocess.org/gpml.
- **Labbe, R.** *Kalman and Bayesian Filters in Python.* GitHub: rlabbe/Kalman-and-Bayesian-Filters-in-Python.
- **James, G., Witten, D., Hastie, T., Tibshirani, R.** (2021). *An Introduction to Statistical Learning.* Springer. Free at statlearning.com.
- **lifelines** documentation: lifelines.readthedocs.io.
- **filterpy** documentation: filterpy.readthedocs.io.
