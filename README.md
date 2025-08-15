# 📊 Policy Response Time Predictor: Forecasting RTI and Grievance Resolution Delays Using Machine Learning 

# Submitted By: 
# 21001012024 
# Suhani Bamania 
# IGDTUW

A comprehensive case analysis engine that predicts resolution times for government policy complaints, appeals, and information requests using machine learning.

---

## 🚀 Features
- Predicts **resolution days** for RTI cases using a regression model.
- Classifies **risk level** and **priority**.
- Detects **season** based on filing month.
- Calculates **confidence score** for predictions.
- Performs **anomaly detection** if actual resolution days are provided.
- Web interface for input via HTML form.


## 📈 Model Performance
- After training, the evaluation script provides:
- Mean Absolute Error (MAE): Average prediction error in days
- R-squared Score: Percentage of variance explained by the model
- Typical performance:
- MAE: 5-8 days
- R²: 0.65-0.80


## 🔍 Understanding the Predictions
*Risk Levels* 

- Low Risk: Standard processing expected (< 20 days)
- Medium Risk: May require additional attention (20-35 days)
- High Risk: Complex case needing priority handling (> 35 days)

*Priority Assignments*

- High: VIP cases, appeals, repeat complaints, or urgent keywords detected
- Medium: Moderate complexity or medium timeline cases
- Low: Standard information requests with simple requirements

*Anomaly Detection States*

- Normal: Standard processing expected
- High Delay Risk: Unusually long timeline predicted
- Fast Track: Case flagged for expedited processing
- Priority Alert: Urgent keywords or special conditions detected.

---

