# Comprehensive Viva Q&A Guide: Energy Demand Forecaster Using RNN, LSTM, and GRU

This document contains **25 high-yield viva / defense examination questions** with detailed, concise answers covering recurrent mathematics, gradient dynamics, data engineering, validation rigor, and deployment choices implemented in **THIS** project.

---

### Q1: Why does SimpleRNN fail to outperform Linear Regression (Ridge) on the Spanish energy dataset?
**Answer**:  
SimpleRNN suffers from the **vanishing gradient problem** during Backpropagation Through Time (BPTT). Over a 24-hour sequence window, the gradient of the loss with respect to early hidden states decays exponentially by a factor of $\|\operatorname{diag}(1 - \tanh^2) W_{hh}^T\|^{24} \approx 0.75^{24} \approx 10^{-3}$. Consequently, the vanilla RNN cannot effectively learn the crucial 24-hour diurnal relationship (same hour yesterday). In contrast, Ridge regression directly receives explicit lag features ($y_{t-24}, y_{t-168}$), giving it a direct linear pathway to the most predictive historical steps without vanishing gradient attenuation.

---

### Q2: Derive mathematically why gradients vanish in SimpleRNN.
**Answer**:  
In SimpleRNN, the hidden state update is $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.  
By the chain rule, the gradient of loss $\mathcal{L}_T$ at time $T$ w.r.t. hidden state $h_k$ ($k < T$) is:
$$\frac{\partial \mathcal{L}_T}{\partial h_k} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{j=k+1}^T \frac{\partial h_j}{\partial h_{j-1}}$$
The Jacobian is $\frac{\partial h_j}{\partial h_{j-1}} = \operatorname{diag}\left(1 - \tanh^2(a_j)\right) W_{hh}^T$.  
Since the derivative of hyperbolic tangent satisfies $0 < 1 - \tanh^2(z) \le 1$, if the spectral radius (maximum singular value) of the weight matrix $\|W_{hh}\| < 1$, the continuous matrix multiplication causes the norm of the gradient to decay exponentially to 0 as $T - k$ increases:
$$\left\| \frac{\partial \mathcal{L}_T}{\partial h_k} \right\| \le C \cdot |\lambda_{\max}(W_{hh})|^{T-k} \xrightarrow{T-k \to \infty} 0$$

---

### Q3: How does LSTM’s Constant Error Carousel (CEC) solve the vanishing gradient problem?
**Answer**:  
LSTM introduces an isolated internal **cell state** $C_t$ governed by an **additive update rule**:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
When computing the gradient through the cell state:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$
Because this connection is additive rather than a matrix multiplication through a squashing activation function, if the forget gate $f_t \approx 1$, the gradient $\frac{\partial \mathcal{L}}{\partial C_k}$ flows backwards across hundreds of time steps with virtually **zero attenuation**. This constant gradient flow is termed the **Constant Error Carousel (CEC)**.

---

### Q4: What is the architectural difference between LSTM and GRU, and why did GRU win in our benchmarks?
**Answer**:  
1. **Gate Count**: LSTM has 3 gates (Forget, Input, Output) and 2 separate state vectors (Cell state $C_t$ and Hidden state $h_t$). GRU has only 2 gates (Reset $r_t$ and Update $z_t$) and a single hidden state $h_t$.
2. **State Transition**: GRU uses a convex linear combination: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.
3. **Why GRU Won**: GRU has **25% fewer trainable parameters** (22,081 vs 29,185 in our 64-32 stacked models). On an hourly dataset with relatively clean, stationary 24-hour cycles, the reduced parameter count provided better regularization against overfitting, lower variance across seeds ($\pm 11.2$ vs $\pm 12.8$ MW), and 15% faster training throughput.

---

### Q5: Why are Sigmoid and Tanh activations chosen for recurrent cells instead of ReLU?
**Answer**:  
- **Sigmoid ($\sigma$) for Gates**: Gates act as continuous information valves between 0 (completely closed/forget) and 1 (completely open/pass). Sigmoid’s output range $[0, 1]$ naturally matches this biological/physical gating requirement.
- **Tanh for State Candidates**: Tanh outputs values in $[-1, 1]$ and is zero-centered. This allows the network to both **increase** (positive state innovation) and **decrease** (negative state inhibition) the memory bank.
- **Why not ReLU inside recurrent loops?**: In SimpleRNN, repeatedly multiplying unbounded ReLU activations ($z \ge 0$) through $W_{hh}$ over 24 time steps almost always triggers catastrophic **exploding activations and exploding gradients** unless weights are initialized strictly to the identity matrix.

---

### Q6: What is temporal data leakage, and how did this project guarantee its complete absence?
**Answer**:  
Data leakage occurs when information from the future (or the test set) is inadvertently used to train the model or compute feature statistics.  
In this project, zero leakage is guaranteed through three strict protocols:
1. **Chronological Splitting**: Enforced strict temporal ordering (Train: 2015-2017 $\to$ Val: late 2017 to early 2018 $\to$ Test: mid-to-late 2018). No random splitting or shuffling.
2. **Train-Only Scaler Fitting**: `StandardScaler.fit()` was called **strictly on the Train partition**. The resulting $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ were used to transform Val and Test without recalculating.
3. **Lag Shifting**: Rolling means and lags were created using `df[col].shift(1)` so that time step $t$ only ever observes data from $\{t-1, t-2, \dots\}$. Unit tests in `tests/test_leakage.py` verify that no test timestamp exists in any training window.

---

### Q7: Why can't we use standard k-fold cross-validation for time series forecasting?
**Answer**:  
Standard k-fold cross-validation randomly samples folds across the entire dataset. In time series, this violates the fundamental arrow of time: training folds would contain data points from Wednesday and Friday to predict Thursday. This creates lookahead bias, artificially inflates validation metrics, and produces models that collapse in true forward-looking production environments. Instead, one must use chronological splits or rolling-origin time-series cross-validation (walk-forward validation).

---

### Q8: Why do we represent temporal features using cyclical Sine and Cosine encodings?
**Answer**:  
If hour of the day is represented as an integer from $0$ to $23$, the numerical distance between 23:00 (11 PM) and 00:00 (midnight) is $23 - 0 = 23$. However, in physical reality, 23:00 and 00:00 are separated by only 1 hour!  
By projecting hours onto a 2D unit circle via:
$$x_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
the Euclidean distance between 23:00 and 00:00 is smoothly preserved:
$$\sqrt{(\sin(23) - \sin(0))^2 + (\cos(23) - \cos(0))^2} \approx 0.26$$
Both sine and cosine are required because a single sine or cosine value is symmetric and ambiguous (e.g., $\sin(2\pi \cdot 6 / 24) = \sin(2\pi \cdot 18 / 24) = 1$). Together, $(\sin, \cos)$ uniquely identify each hour.

---

### Q9: What is the difference between Direct multi-step and Recursive multi-step forecasting?
**Answer**:  
- **Recursive Strategy**: A single-step model ($\hat{y}_{t+1} = f(y_t, y_{t-1}, \dots)$) is called iteratively. To predict $t+2$, the model feeds its own prediction $\hat{y}_{t+1}$ as an input. This causes compounding prediction error: any mistake at step 1 distorts step 2, causing runaway error divergence over 24 hours.
- **Direct Strategy**: The neural network outputs all 24 future time steps simultaneously via a Dense(24) layer ($\hat{\mathbf{y}}_{t+1:t+24} = f(\mathbf{x}_t)$). The loss function optimizes all 24 horizons concurrently. In our tests, Direct forecasting achieved $1,085.40$ MW RMSE versus $1,420.80$ MW for Recursive rollout.

---

### Q10: How does the From-Scratch Genetic Algorithm tuner work in this project?
**Answer**:  
The Genetic Algorithm (`src/training/tune_genetic.py`) treats a hyperparameter set as a **chromosome**:
- **Genes**: `window_size` (12, 24, 48), `units_l1` (32, 64, 96), `units_l2` (16, 32, 64), `dropout` (0.1, 0.2, 0.3), `learning_rate` (0.0005, 0.001, 0.002), `batch_size` (32, 64).
- **Fitness Function**: Validation RMSE after early-stopped training (lower is better).
- **Tournament Selection**: Contestants are sampled randomly ($k=2$); the individual with the lowest validation RMSE wins reproduction rights.
- **Uniform Crossover**: Two parent chromosomes swap genes with probability $0.8$.
- **Mutation**: Randomly perturbs genes with probability $0.25$ to maintain genetic diversity and escape local minima.
- **Elitism**: The single best individual of each generation survives intact into the next generation.

---

### Q11: How does Bayesian Optimization (Optuna) differ from the Genetic Algorithm?
**Answer**:  
- **Genetic Algorithms** are heuristic, population-based evolutionary algorithms that operate without an explicit mathematical surrogate model; they explore combinations via stochastic crossover and mutation.
- **Optuna (Tree-structured Parzen Estimator - TPE)** is a Bayesian optimization method that constructs two probability density functions: $\ell(x) = p(x \mid y < y^*)$ (good configurations) and $g(x) = p(x \mid y \ge y^*)$ (poor configurations). It selects the next trial by maximizing the Expected Improvement $\frac{\ell(x)}{g(x)}$.  
*Empirical finding*: Optuna converged ~2.1x faster than the GA on our search space.

---

### Q12: Why do we use EarlyStopping with `restore_best_weights=True`?
**Answer**:  
Without `restore_best_weights=True`, training stops when the validation loss fails to improve after $N$ patience epochs, leaving the model with weights from the *worst* overfitted epoch ($t_{\text{best}} + N$). Setting `restore_best_weights=True` acts as an optimal regularizer, guaranteeing that the model state rolled back to the exact epoch where validation loss achieved its global minimum.

---

### Q13: What is the difference between Batch Gradient Descent, SGD, and Mini-Batch GD? Why batch size 64?
**Answer**:  
- **Batch GD**: Computes gradients over the entire training set (24,528 samples). Computationally prohibitive, deterministic, and easily trapped in shallow local minima.
- **Stochastic GD (SGD)**: Updates weights sample-by-sample. Highly noisy trajectory; does not vectorize on CPU/GPU hardware.
- **Mini-Batch GD (Used: Batch Size 64)**: Balances computational vectorization efficiency with stochastic regularization. Batch size 64 fits comfortably in CPU L3 cache and provides sufficient gradient variance to escape saddle points.

---

### Q14: How does L2 Regularization differ from Dropout in recurrent architectures?
**Answer**:  
- **L2 Regularization (Weight Decay)**: Adds a penalty $\frac{\lambda}{2} \sum w_i^2$ directly to the loss function. It shrinks weight magnitudes towards zero, encouraging smooth decision boundaries and preventing any single feature from dominating.
- **Dropout (Rate 0.2)**: Randomly zeroes out 20% of activation units during each forward pass. This forces neurons to develop redundant, co-adapted representations rather than relying on specific feature memorization. In our tests, Dropout (0.2) + EarlyStopping outperformed L2 regularization alone.

---

### Q15: Explain Pinball Loss (Quantile Loss) mathematically and how it yields prediction intervals.
**Answer**:  
Standard MSE optimizes for the conditional **mean** $\mathbb{E}[y \mid x]$.  
**Pinball Loss** optimizes for an arbitrary target quantile $\tau \in (0, 1)$:
$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} 
\tau (y - \hat{y}) & \text{if } y \ge \hat{y} \\ 
(1 - \tau) (\hat{y} - y) & \text{if } y < \hat{y} 
\end{cases}$$
- For $\tau = 0.50$, it penalizes positive and negative errors symmetrically, learning the conditional **median**.
- For $\tau = 0.90$, under-forecasting ($y \ge \hat{y}$) is penalized $9\times$ more severely than over-forecasting, pushing predictions to the 90th percentile.
- Training three heads simultaneously at $\tau \in \{0.10, 0.50, 0.90\}$ yields a calibrated 80% prediction interval $[P_{10}, P_{90}]$.

---

### Q16: What does an 81.4% empirical coverage for a nominal 80% prediction interval signify?
**Answer**:  
It signifies excellent statistical calibration. Across 5,256 unseen test hours, 81.4% of actual demand values fell between the predicted P10 and P90 bounds. Because $81.4\% \approx 80\%$, the uncertainty bounds are neither excessively wide (over-conservative) nor excessively narrow (under-confident).

---

### Q17: How does the Transformer Multi-Head Self-Attention model compare to LSTM on sequence data?
**Answer**:  
- **LSTM**: Sequential recurrence $\mathcal{O}(T)$ operations. Cannot parallelize across time steps; relies on gating to pass information sequentially.
- **Transformer**: Computes self-attention across all time steps simultaneously:
  $$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  Every time step attends directly to every other time step in $\mathcal{O}(1)$ sequential path length. On our 24h window, the Transformer achieved an RMSE of $755.20$ MW (matching LSTM), with attention weights revealing strong correlation between $t$ and $t-24$.

---

### Q18: Why do Transformers require Positional Encoding while RNNs do not?
**Answer**:  
Recurrent neural networks process tokens sequentially ($t=1$, then $t=2$, then $t=3$), inherently embedding temporal order into the hidden state transition $h_t = f(h_{t-1}, x_t)$.  
In contrast, Transformer self-attention is **permutation-invariant**: swapping time step 3 and time step 21 produces identical attention score sets. Therefore, sinusoidal positional encodings ($\sin / \cos$ wave functions of varying frequencies) must be added directly to the input vectors to explicitly inject chronological sequence order.

---

### Q19: What is Permutation Feature Importance, and which features were most influential in our models?
**Answer**:  
Permutation feature importance evaluates the increase in test RMSE when the values of a single feature column are randomly shuffled, breaking its relationship with the target while maintaining the overall feature distribution.  
*Rankings in our model*:
1. **Target Lag 24h**: $\Delta \text{RMSE} = +312.4$ MW (strongest predictor)
2. **Mean Temperature**: $\Delta \text{RMSE} = +84.2$ MW (critical for cooling/heating loads)
3. **Hour Sin/Cos**: $\Delta \text{RMSE} = +62.8$ MW
4. **Is Weekend**: $\Delta \text{RMSE} = +41.5$ MW
5. **Wind Speed / Humidity**: $\Delta \text{RMSE} \le +12.0$ MW

---

### Q20: What is Concept Drift in energy forecasting, and how does `drift.py` detect it?
**Answer**:  
Concept drift occurs when the statistical relationship between input features and target load changes over time: $P_t(y \mid x) \neq P_{t+1}(y \mid x)$. This can be triggered by sudden industrial restructuring, prolonged heatwaves, or energy tariff shocks.  
Our `drift.py` module continuously monitors model health by computing the rolling 7-day RMSE ($168$ hours):
$$\text{Drift Ratio} = \frac{\text{RMSE}_{\text{rolling 7d}}}{\text{RMSE}_{\text{validation baseline}}}$$
If the drift ratio exceeds the threshold of $1.25$ (a 25% performance degradation), a `DRIFT_ALERT` is raised and an automated retrain trigger (`make retrain`) is executed.

---

### Q21: Why did we evaluate neural models across 5 distinct random seeds (`42, 123, 456, 789, 999`)?
**Answer**:  
Neural network training is inherently stochastic due to random weight initialization (Glorot/Xavier uniform) and mini-batch shuffle ordering. A single run might achieve a favorable score purely due to lucky initialization. Evaluating across 5 seeds and reporting **Mean $\pm$ Standard Deviation** provides statistical confidence that GRU’s outperformance ($748.90 \pm 11.2$ MW) is statistically significant and reproducible rather than random noise.

---

### Q22: What are the regression metrics evaluated in this project, and why is RMSE preferred over MSE or MAE for power grids?
**Answer**:  
- **MAE** ($\frac{1}{N} \sum |y - \hat{y}|$): Linear error penalty; easily interpretable in Megawatts.
- **MSE** ($\frac{1}{N} \sum (y - \hat{y})^2$): Penalizes large errors quadratically, but units are in $\text{MW}^2$.
- **RMSE** ($\sqrt{\text{MSE}}$): Preserves the quadratic penalty on large errors while restoring units to **Megawatts (MW)**.  
*Why RMSE is preferred in grid management*: A single 2,000 MW error is drastically more hazardous to grid frequency stability than two separate 1,000 MW errors. Quadratic penalization severely discourages extreme outliers.
- **MAPE** ($\frac{1}{N} \sum \left|\frac{y - \hat{y}}{y}\right| \times 100\%$): Scale-independent percentage metric for cross-region comparison.

---

### Q23: What are the main failure modes of our forecaster?
**Answer**:  
1. **Unusual Holiday Transitions**: Bridge holidays (*puentes* in Spain), where a Tuesday holiday causes Monday to experience weekend-like industrial shutdown, leading to over-forecasting.
2. **Extreme Heatwave Anomalies**: Days where temperatures exceed historical training distributions (> 42°C in Seville), resulting in under-forecasting due to unprecedented air conditioning demand.
3. **Rapid Frontal Weather Changes**: Sudden unpredicted cloud cover or localized thunderstorms causing sudden daytime lighting and temperature shifts.

---

### Q24: What is the economic impact of a 1% reduction in MAPE for a national grid?
**Answer**:  
According to energy economics literature (e.g., Hobbs et al., IEEE Transactions on Power Systems), for a mid-sized national transmission grid like Spain (average load ~28,000 MW), a **1% reduction in day-ahead forecast MAPE saves approximately €10M to €20M annually** in avoided spinning reserve capacity, reduced spot-market balancing penalties, and optimized thermal generator dispatch.

---

### Q25: How would you extend this project into an enterprise production environment?
**Answer**:  
1. **Real-Time Data Streaming**: Ingest live SCADA and Open-Meteo weather forecasts via Apache Kafka / Redpanda.
2. **Hierarchical Reconciliation**: Forecast regional substations and aggregate upward using MinT (Minimum Trace) hierarchical reconciliation to guarantee that regional forecasts sum exactly to national demand.
3. **Automated CI/CD**: Deploy model via GitHub Actions to an Amazon EKS / Google Kubernetes Engine cluster with blue-green canary deployments and shadow inference monitoring.
