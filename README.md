# Craigslist-used-car-predictor
Predicting Craigslist used car prices : A comparative analysis between a baseline model and an enhanced model using LLM-generated MSRP features

## Original Dataset Information

This project utilizes the **[Craigslist Used Car Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)** provided by Kaggle for analysis and predictive modeling.

- **Source:** Kaggle (Craigslist Used Vehicle Dataset)
- **Dataset Size:** Approx. 426,880 records and 26 features (Raw data)
- **Key Features:**
  - **Target Variable:** `price`
  - **Vehicle Specifications:** `year`, `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `odometer`, `transmission`, `drive`, `type`, `paint_color`, etc.
  - **Metadata:** `id`, `url`, `posting_date`, etc.
- **Data Characteristics:** This dataset contains actual used car listings scraped from Craigslist, the largest classified advertisements website in the US. It inherently includes a significant amount of missing values, extreme outliers (e.g., fake listings), and highly subjective text descriptions. Representing typical **'noisy real-world data'**, it serves as an excellent foundation to demonstrate and validate advanced data preprocessing and feature engineering capabilities.