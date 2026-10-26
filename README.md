# Time Series and Trend Analysis in Internet Data
## Report: Activity Forecasting, Anomaly Detection, and Viral Spike Monitoring

**Course:** Internet Data Analysis for Master's Students
**Module:** 2, Lecture 7
**Student:** [Your Name]
**Date:** [Date]

---

## 1. Introduction

### 1.1 Objective

The objective of this report is to apply classical and modern time series analysis methods to an internet data time series in order to accomplish three tasks:

1. **Forecasting** — Use Prophet to produce a 48-hour-ahead forecast and evaluate its accuracy (MAE, MAPE).
2. **Anomaly Detection** — Implement and compare three anomaly detection methods (Z-score, Isolation Forest, Prophet anomalies).
3. **Viral Spike Monitoring** — Build a real-time viral spike monitor and test it on historical data.

### 1.2 Dataset

- **Data source:** [Source of your data — e.g., Twitter/X posts, Reddit comments, Google Trends, synthetic data]
- **Time period:** [e.g., January 1–30, 2024 — 30 days]
- **Aggregation:** Hourly (`1H`)
- **Total observations:** [e.g., 720 hourly points]
- **Variable:** [e.g., number of posts per hour]

> **[Insert a short description of your dataset here — what the data is, how it was obtained, and why it was chosen.]**

---

## 2. Methodology

### 2.1 Data Preprocessing

Internet data events do not occur at regular time intervals (irregular frequency). Therefore, the first step is to aggregate the data into **regular time intervals**.

```python
def aggregate_to_regular_intervals(posts_df, freq='1H'):
    posts_df['timestamp'] = pd.to_datetime(posts_df['timestamp'])
    posts_df = posts_df.set_index('timestamp')
    activity = posts_df.resample(freq).size()
    activity = activity.asfreq(freq, fill_value=0)
    return activity
```

Missing intervals are filled with **zero**.

### 2.2 Decomposition

To separate trend, seasonality, and residual components, **`seasonal_decompose`** (additive model, period=24) is used.

```python
def decompose_activity(activity_series, period=24):
    decomposition = seasonal_decompose(activity_series, model='additive', period=period)
    return {'trend': ..., 'seasonal': ..., 'residual': ...}
```

For multiple seasonality (daily + weekly), **Prophet** is used, as it natively supports daily, weekly, and yearly seasonality simultaneously.

### 2.3 Forecasting with Prophet

```python
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    changepoint_prior_scale=0.05
)
model.add_country_holidays(country_name='US')
model.fit(df)
forecast = model.predict(future)
```

- `changepoint_prior_scale=0.05` controls sensitivity to trend changes.
- `add_country_holidays` accounts for the effect of holidays.

### 2.4 Anomaly Detection Methods (3 Methods)

| Method | Principle | Strengths | Weaknesses |
|---|---|---|---|
| **Z-score** | Compare against rolling window (24) mean/std; flag \|z\| > 3 | Simple, fast | Captures only linear patterns |
| **Isolation Forest** | Isolate anomalies using random forests | Captures non-linear patterns | Requires parameter tuning |
| **Prophet anomalies** | Flag values outside `yhat_lower`/`yhat_upper` | Incorporates seasonality | Requires a well-fitted model |

### 2.5 Viral Spike Detection

```python
def detect_viral_spike(activity_series, growth_threshold=3.0):
    baseline = activity_series.rolling(window=24*7).median()
    growth_ratio = activity_series / (baseline + 1)
    spikes = growth_ratio[growth_ratio > growth_threshold]
    ...
```

The baseline is computed as the **one-week median**. A spike is flagged when current activity exceeds three times the baseline. Severity is assigned as:
- `critical` — ratio > 10
- `high` — ratio > 5
- `moderate` — ratio > 3

---

## 3. Results

> **[⚠️ This section must be filled with the actual numbers obtained from your own run. The values below are placeholders only.]**

### 3.1 Decomposition Results

[Insert decomposition plot here — trend, seasonal, residual]

**Observations:**
- The trend shows [e.g., a gradual increase over the 30-day period].
- Daily seasonality peaks between [e.g., 6 PM and 9 PM].
- Weekly seasonality shows lower activity on [e.g., Saturdays and Sundays].

### 3.2 Assignment 1: Forecast Accuracy

| Metric | Value |
|---|---|
| **MAE** | [insert your result] |
| **MAPE** | [insert your result] % |
| Holdout period | 48 hours |

[Insert forecast plot here — actual vs predicted, with confidence interval]

**Interpretation:** [e.g., MAE is approximately 15% of the mean activity, indicating the forecast is within an acceptable range.]

### 3.3 Assignment 2: Anomaly Detection Comparison

| Method | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Z-score | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Isolation Forest | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Prophet anomalies | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

[Insert comparison plot here]

**Interpretation:** [e.g., Isolation Forest achieves higher precision, while Z-score achieves better recall. Prophet anomalies, by incorporating seasonality, reduces false positives caused by daily patterns.]

### 3.4 Assignment 3: Viral Spike Monitor

| Severity | Number of Alerts | Example Timestamp |
|---|---|---|
| Critical | [ ] | [ ] |
| High | [ ] | [ ] |
| Moderate | [ ] | [ ] |

[Insert spike plot here]

**Interpretation:** [e.g., the monitor successfully detected the synthetic viral spike on day 20. The alert was raised within one hour of the spike onset, making it suitable for early detection.]

---

## 4. Discussion

### 4.1 Method Comparison

[e.g., Prophet handles multiple seasonality well and is therefore the most suitable for forecasting. However, it may overfit when data is limited. For anomaly detection, Isolation Forest captures non-linear patterns but requires parameter tuning.]

### 4.2 Limitations

- [e.g., With only 30 days of data, yearly seasonality cannot be captured.]
- [e.g., A synthetic spike was used, so results may not fully generalize to real-world viral events.]
- [e.g., Z-score results depend heavily on the rolling window size.]

### 4.3 Practical Implications

[e.g., This monitor can be applied to brand reputation monitoring. The alert threshold can be adjusted according to business requirements.]

---

## 5. Conclusion

In this report, on an internet data time series, we:

1. Used **Prophet** to produce a 48-hour-ahead forecast (MAE = [ ], MAPE = [ ]%).
2. Compared **three anomaly detection methods** and found that [best method] performed best for this dataset.
3. Built a **real-time viral spike monitor** that successfully detected historical spikes.

Future improvements include using a longer data period, incorporating external events (holidays, news), and comparing against deep learning models (LSTM).

---

## 6. Appendix: Code

**Notebook link:** [Insert your Colab link here]

**Or GitHub repository:** [link]

---

## Checklist Before Submission

- [ ] Your name and date are filled in.
- [ ] Data source and time period are clearly stated.
- [ ] MAE and MAPE values are filled with actual results.
- [ ] Comparison table includes all of TP/FP/FN/Precision/Recall/F1.
- [ ] Plots are inserted (decomposition, forecast, anomaly, spike).
- [ ] Discussion explains which method performed best and why.
- [ ] Limitations are stated.
- [ ] Colab/GitHub link is included.
- [ ] The report has been converted to PDF and reviewed.
