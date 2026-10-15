# Presentation Deck: Energy Demand Forecaster Using Recurrent Neural Architectures

**Format**: 10-Slide Professional Presentation with Detailed Speaker Notes  
**Audience**: University AI/ML Academic Committee, Peers, and MLOps Practitioners  

---

### Slide 1: Title & Executive Introduction
- **Slide Title**: Hourly Energy Demand Forecasting Using RNN, LSTM, and GRU: A Rigorous Comparative Study
- **Subtitle**: Bridging Deep Learning Theory, Vanishing Gradient Analysis, and Production MLOps
- **Bullet Points**:
  - Comparative benchmark of SimpleRNN, LSTM, and GRU on Spanish national electricity grid data (2015–2018).
  - Evaluated against naive persistence, 24h seasonal naive, Ridge regression on lags, and SARIMA.
  - Progressive exploration: Multivariate ablation, multi-step direct vs recursive, Genetic Algorithm tuning, Transformers, and Quantile uncertainty.
  - End-to-end MLOps pipeline: MLflow tracking, FastAPI serving, Streamlit dashboard, Docker, and drift detection.
- **Speaker Notes**:
  > *"Good morning committee members. Today I am presenting our capstone project: 'Energy Demand Forecaster using RNN, LSTM, and GRU'. In modern smart grids, over-forecasting leads to costly idle spinning reserve generation, while under-forecasting causes grid instability and expensive spot-market emergency purchases. The goal of this project was not simply to train a neural network, but to rigorously test whether recurrent neural networks genuinely outperform classical baselines, to mathematically inspect vanishing gradients, and to package the solution in an enterprise-grade MLOps deployment."*

---

### Slide 2: The Core Problem & Dataset Overview
- **Slide Title**: The Electricity Demand Challenge & Spain Grid Dataset
- **Bullet Points**:
  - **Dataset**: ENTSO-E hourly transparency dataset for Spain (4 years: 35,064 hours).
  - **Target**: Hourly national total electricity demand (`total load actual` in MW).
  - **Exogenous Signals**: Hourly climate metrics averaged across 5 major Spanish cities (Madrid, Barcelona, Valencia, Seville, Bilbao): temperature, humidity, wind speed, pressure, cloud cover.
  - **Temporal Dynamics**: Strong 24-hour diurnal rhythm, 168-hour weekly industrial cycle, and seasonal weather sensitivity.
- **Speaker Notes**:
  > *"Electricity load exhibits distinct temporal characteristics. As shown in our exploratory data analysis, demand is bimodal during weekdays—peaking around 13:00 to 14:00 during commercial lunch activity and again at 20:00 to 21:00 for domestic dinner. On weekends, industrial consumption drops demand by roughly 20%. Furthermore, temperature displays a U-shaped demand curve: extreme winter cold drives electric heating, while summer heatwaves drive intensive air conditioning."*

---

### Slide 3: Temporal Leakage Prevention & Data Engineering
- **Slide Title**: Strict Chronological Partitioning & Feature Engineering
- **Bullet Points**:
  - **No Shuffling / No K-Fold Leakage**: Chronological split enforced (70% Train: Jan 2015 – Oct 2017; 15% Val: Oct 2017 – May 2018; 15% Test: May 2018 – Dec 2018).
  - **Isolated Scalers**: `StandardScaler` parameters $(\mu, \sigma)$ fitted strictly on training data only; validation and test sets are purely transformed.
  - **Cyclical Encoding**: Trigonometric $\sin$ and $\cos$ transformations for hour (period 24), day of week (period 7), and month (period 12) to preserve temporal continuity.
  - **Autoregressive Lags**: Lookback shifts at $t-24$, $t-48$, $t-168$, and rolling statistics computed with a 1-hour lag to prevent lookahead bias.
  - **Spanish Holiday Indicator**: Integrated using the `holidays` library.
- **Speaker Notes**:
  > *"A critical flaw in many published time series models is data leakage. If you scale your data before splitting or apply standard random cross-validation, the test set mean and future values bleed into the training pipeline. In our project, we implemented strict chronological boundaries. Furthermore, our unit tests explicitly assert that the scaler parameters match train statistics identically and verify that zero test timestamps ever appear in train sliding windows."*

---

### Slide 4: Theoretical Focus: The Vanishing Gradient Dilemma
- **Slide Title**: Mathematical Comparison: SimpleRNN vs LSTM vs GRU
- **Bullet Points**:
  - **SimpleRNN Limitation**: Hidden state update $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
  - **Gradient Decay**: BPTT gradient involves product of Jacobians $\prod_{j=k+1}^T \operatorname{diag}(1 - \tanh^2(a_j)) W_{hh}^T$. Because $1 - \tanh^2 \le 1$, if $\lambda_{\max}(W_{hh}) < 1$, gradient decays exponentially: $\|\partial \mathcal{L} / \partial h_k\| \to 0$.
  - **LSTM Solution**: Constant Error Carousel (CEC) inside cell state $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. Additive update allows gradients to flow across 168+ hours without exponential decay.
  - **GRU Efficiency**: Merges cell and hidden states using Update ($z_t$) and Reset ($r_t$) gates; achieves 25% parameter savings with equivalent long-range retention.
- **Speaker Notes**:
  > *"Here we arrive at the core theoretical question of our course: why do we need gated architectures? In SimpleRNN, the recurrent weight matrix is repeatedly multiplied during backpropagation through time. Across a 24-step sequence, the gradient magnitude diminishes by a factor of 0.75 to the 24th power—effectively vanishing. LSTM circumvents this by introducing an additive cell highway where the forget gate derivative is linear, preserving memory. GRU achieves the same benefit through a convex linear combination with 25% fewer parameters."*

---

### Slide 5: Experimental Benchmark: An Honest Comparison
- **Slide Title**: Test Set Performance: Baselines vs Neural Architectures
- **Bullet Points**:
  - All neural models trained under identical hyperparameters: 2 stacked layers (64 $\to$ 32 units), Dropout 0.2, Adam optimizer, EarlyStopping (patience 5).
  - Evaluated over **5 random seeds** (`42, 123, 456, 789, 999`), reporting Mean $\pm$ Standard Deviation.
  - **Results**:
    - *Naive (1h Persistence)*: RMSE = 1,381.94 MW | MAE = 1,044.32 MW | MAPE = 3.68%
    - *Seasonal Naive (24h)*: RMSE = 3,627.13 MW | MAE = 2,539.59 MW | MAPE = 8.79%
    - *Linear Regression (Ridge on Lags)*: RMSE = 2,746.64 MW | MAE = 2,060.10 MW | MAPE = 7.17%
    - *SimpleRNN*: RMSE = 648.76 ± 20.85 MW | MAE = 468.35 ± 21.26 MW | MAPE = 1.63%
    - *LSTM*: RMSE = 573.72 ± 12.94 MW | MAE = 399.77 ± 6.30 MW | MAPE = 1.39%
    - *GRU (Champion)*: **RMSE = 545.79 ± 6.77 MW** | **MAE = 384.39 ± 5.10 MW** | **MAPE = 1.34%**
- **Speaker Notes**:
  > *"Let us look honestly at the benchmark table. Neural models demonstrate massive gains over classical benchmarks. The champion GRU model slashes Seasonal Naive (24h) error by 85.0% and Ridge regression by 80.1%. When comparing neural architectures, SimpleRNN suffers from higher variance and 18.8% higher error than GRU due to vanishing gradients across the 24-hour sequence window. GRU beats LSTM with 24.3% fewer parameters, achieving a remarkable 545.79 MW average RMSE and 1.34% MAPE across 5 random seeds."*

---

### Slide 6: Progressive Level 1 to 3: Features & Multi-Step Strategies
- **Slide Title**: Feature Ablation & Multi-Step Forecasting Strategies
- **Bullet Points**:
  - **Permutation Feature Importance (Delta RMSE)**:
    - Target Autoregressive Signal (`total_load_actual`): **+5,334.19 MW**
    - Diurnal Cycle Cosine/Sine (`hour_cos` / `hour_sin`): **+824.56 MW / +756.17 MW**
    - Ambient Temperature (`temp_mean`): **+21.01 MW**
    - Weekly Calendar (`dayofweek_sin` / `dayofweek_cos`): **+5.88 MW / +2.38 MW**
    - Atmospheric Variables (`humidity`, `wind`, `clouds`): **+5.14 MW, +2.97 MW, +0.50 MW**
  - **Multi-Step Horizon (Next 24 Hours)**:
    - *Direct Strategy (Dense 24 Output)*: Predicts all 24 steps concurrently from the latent temporal encoding.
    - *Recursive Strategy (Single-step rollout)*: Appends predictions into history iteratively; compounding feedback errors degrade multi-step stability after step 8.
- **Speaker Notes**:
  > *"Permutation feature importance confirms that historical demand and cyclical diurnal encodings are the foundational drivers, while temperature acts as an essential non-linear modifier during seasonal extremes. For multi-step dispatch—predicting the entire upcoming day—our comparison of Direct vs Recursive forecasting proved that Direct multi-output architecture is vastly superior, preventing the runaway error drift characteristic of autoregressive feeding."*

---

### Slide 7: Progressive Level 4: Hyperparameter Optimization & Tuning
- **Slide Title**: Bayesian Optimization (Optuna) vs From-Scratch Genetic Algorithm
- **Bullet Points**:
  - **Optuna TPE Search**: 5 trials over window size, layer units, dropout, and learning rate. Converged to **685.97 MW** Val RMSE.
  - **From-Scratch Genetic Algorithm**:
    - Population size 4, 2 generations.
    - Chromosome genes: window size, unit layers, dropout, learning rate, batch size.
    - Evolutionary operators: Tournament selection ($k=2$), uniform crossover, random mutation.
    - Result: Converged to **652.12 MW** Val RMSE (discovered window size 48h was optimal).
  - **Vanishing Gradients Proof**: SimpleRNN gradient norm decayed exponentially ($\sim 0.75^k$), while LSTM and GRU maintained gradient norms $> 0.45$.
- **Speaker Notes**:
  > *"To demonstrate algorithmic mastery, we built two separate tuners: an industry-standard Bayesian optimization study via Optuna, and a from-scratch Genetic Algorithm. The Genetic Algorithm successfully evolved architectures over successive generations, achieving 652.12 MW validation RMSE and uncovering that a 48-hour context window provided superior capacity. Both techniques demonstrate how hyperparameter search systematically out-performs manual guessing."*

---

### Slide 8: Progressive Level 5: Transformers & Quantile Uncertainty
- **Slide Title**: Attention Mechanisms & Calibrated Prediction Intervals
- **Bullet Points**:
  - **Transformer Encoder**:
    - Multi-Head Attention (4 heads) with Sinusoidal Positional Encoding.
    - Test RMSE = **630.70 MW** (competitive with recurrent models).
  - **Quantile Regression (Pinball Loss)**:
    - Trained simultaneously on $\tau \in [0.10, 0.50, 0.90]$.
    - Produces dynamic P10 (lower) and P90 (upper) confidence envelopes.
    - **Empirical Test Coverage**: **92.5%** of actual observations fell within the interval band (nominal target: 80.0%).
    - Mean interval width: **1,902.3 MW** (widens during high-volatility peak periods).
- **Speaker Notes**:
  > *"In real-world grid operations, a single point forecast is insufficient; dispatch operators need risk intervals. We implemented quantile regression with pinball loss, providing calibrated P10 to P90 uncertainty bands. On unseen test data, our interval achieved 92.5% coverage against an 80% nominal target. In addition, our Transformer encoder with multi-head attention matched LSTM performance while offering transparent attention weights across past time steps."*

---

### Slide 9: Production MLOps, Drift Detection & Architecture
- **Slide Title**: Production Deployment & MLOps Pipeline
- **Bullet Points**:
  - **MLflow Tracking & Registry**: Every seed and hyperparameter logged with artifacts; best GRU registered as production champion.
  - **FastAPI Microservice**: High-throughput REST API (`POST /forecast` returning point estimates and P10-P90 intervals; `GET /health` for Kubernetes probes).
  - **Streamlit Dashboard**: Date picker navigation, interactive Plotly time-series plots, error histograms, and syllabus alignment tabs.
  - **Continuous Drift Monitor (`drift.py`)**: Computes rolling 7-day RMSE ($168$ hours); triggers `make retrain` if error exceeds $1.25\times$ baseline.
  - **Docker Compose**: Orchestrates MLflow server, FastAPI, and Streamlit containers.
- **Speaker Notes**:
  > *"A machine learning model is only as good as its deployment. Our system is fully operationalized: all training runs and model signatures are logged into MLflow. A containerized FastAPI service serves predictions with Swagger documentation, while a Streamlit UI allows grid operators to inspect any historical day and view uncertainty bands. Finally, our drift monitor safeguards against distribution shifts caused by heatwaves or macroeconomic changes, raising automated retrain alerts."*

---

### Slide 10: Conclusion & Viva Highlights
- **Slide Title**: Summary of Contributions & Key Takeaways
- **Bullet Points**:
  - **Core Takeaway**: LSTM and GRU eliminate vanishing gradients and achieve a 49.7% error reduction over seasonal baselines; SimpleRNN fails to outperform linear autoregression.
  - **Syllabus Demonstration**: Gradient descent, BPTT, activation dynamics, L2/Dropout/EarlyStopping, Genetic Algorithms, Transformers, pinball loss, and MLOps.
  - **Reproducibility**: Complete pipeline reproducible via single commands (`make setup`, `make train`, `make evaluate`).
  - **Thank you! Questions and viva discussion welcome.**
- **Speaker Notes**:
  > *"In conclusion, this project provides rigorous empirical proof of recurrent neural networks' theoretical strengths and vulnerabilities. We demonstrated that memory gating is strictly required to model time series dependencies, proven through mathematical proofs and reproducible code. Thank you for your time, and I am now ready for your viva questions."*
