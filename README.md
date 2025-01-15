# Climatologically-driven temporal predictive modeling of dengue cases in Philippine locations

## Abstract
![Graphical%20Abstract/Graphical%20Abstract.jpg](Graphical%20Abstract/Graphical%20Abstract.jpg)

Dengue fever is a major public health concern, particularly in countries like the Philippines where dengue incidence is surging and highly variable. To address this, predictive models for dengue cases were developed for Bulacan, Quezon City, and Rizal—Philippine locations within the same dengue case cluster—using statistical and machine learning techniques. The process started with K-means clustering and t-SNE for location grouping, followed by stationarity checks to ensure data stability. The study integrated temporal patterns of dengue incidence, geographical features, outlier features, and selected climatic factors to create predictive features of dengue cases. Univariate correlation checks were then used to reduce dimensionality. For time series forecasting, SARIMA and SARIMAX models tuned by Auto-ARIMA were applied. These statistical models were compared to classical machine learning models obtained through TPOT, with hyperparameters tuned with Optuna. The **Stochastic Gradient Descent (SGD) Regressor** emerged as the best model, achieving a **mean absolute error (MAE) of 32.26, a root mean squared error (RMSE) of 59.32, and an R-squared (R²) value of 83.40%**.  This demonstrated significant predictive accuracy compared to other models. However, the model did not show potential in accurately predicting dengue outbreaks, indicating that more sophisticated features are needed for precise outbreak predictions.

-------
![Assets/ProjectMethodology.png](Assets/ProjectMethodology.png)
The details of the methodology and results will be discussed in the following sections.
- [Dataset Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Processing](#feature-processing)
- [Model Survey](#model-survey)
- [Hyperparameter Tuning](#results-and-discussion)
- [Model Evaluation and Forecast](#model-evaluation-and-forecast)
- [Conclusion and Recommendation](#conclusion-and-recommendation)
- [References](#references)

`--update in progress`
## Data Collection 

## Data Preprocessing

## Feature Processing

## Model Survey

## Hyperparameter Tuning

## Model Evaluation and Forecast

## Conclusion and Recommendation

## References

