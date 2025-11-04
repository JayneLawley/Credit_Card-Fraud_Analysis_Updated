# Project: Credit Card Fraud Analysis

This project uses exploratory data analysis (EDA), statistical testing and predictive modelling to investigate fraud patterns, evaluate trade-offs between detecting fraudulent activity and reducing false alerts, and communicate insights for key business stakeholders.

**Dataset Content**

- Source: Synthetic, publicly available on Kaggle: *Credit Card Fraud Dataset*
- Size: 1,000,000 transactions (8.7% labelled as fraudulent)
- Sample Used: 100,000-row stratified sample (preserves fraud prevalence and feature distributions)
- Sampling Details: Row counts, fraud rate, and random seed documented in the ETL notebook; metadata stored in a JSON file.

**Main variables:**

- distance_from_home: Distance from customer's home to transaction location
- distance_from_last_transaction: Distance between consecutive transactions
- ratio_to_median_purchase_price: Ratio of current transaction to customer's median purchase size
- repeat_retailer: Whether transaction occurred at a previously used retailer
- used_chip: Whether a card chip was used
- used_pin_number: Whether a PIN was used
- fraud: Target variable (1 = fraudulent, 0 = genuine)
---

## Business Rationale

Credit card fraud is a growing global challenge. Recent reports highlight an increase in card fraud losses worldwide, with new tactics exploiting gaps in current defences (FICO, 2023). Fraud has serious costs for all parties: victims suffer financial losses and stress, while banks face direct losses, higher operating costs, and reputational damage.

To combat these threats, financial institutions are increasingly turning to machine learning and AI. Industry experts emphasise that effective fraud prevention requires models that learn from global fraud patterns, analytics that adapt in real time, and platforms that deploy rapidly while amplifying human expertise (FICO, 2023).

In fraud detection, there is a crucial trade-off between catching more fraud and avoiding false alarms. Missing a fraudulent transaction (a false negative) directly causes financial loss, whereas incorrectly flagging a genuine customer (a false positive) wastes investigative effort and frustrates customers (Soto-Valero, 2023). Success therefore is not about raw accuracy, which can be misleading on imbalanced datasets, but about balancing these outcomes.

For this project I used a 100k sample of a synthetic dataset, with 8.7% labelled as fraudulent. While this fraud rate is much higher than in real-world scenarios (often <0.5%), the dataset provides a practical context in which to explore fraud detection methods. Its main strength is that the features are intuitive (e.g. distance from home, chip use, online order), making it possible to build models that can be explained clearly to business stakeholders.

**Business Requirements**

Modern fraud prevention relies on machine learning and AI to detect evolving fraud patterns and optimise investigation workflows. The main challenge is balancing recall (catching more fraud) and precision (avoiding false alerts).

- False negatives: Direct loss
- False positives: Wasted resources, customer dissatisfaction

**Business Goals**

- Communicate insights via a dashboard highlighting trade-offs between detecting fraud and minimising false alerts
- Evaluate model performance using interpretable metrics and threshold tuning
- Demonstrate how analytics inform real-world fraud management strategies

---

## Exploratory Data Analysis and Testing Methodology

Exploratory Data Analysis (EDA) was used to examine feature distributions, validate statistical tests, and identify fraud patterns across both continuous and categorical variables. The analysis combined visual exploration with formal testing to ensure that findings were both interpretable and statistically sound.

**Main Data Analysis Libraries**
- Pandas (for data loading, cleaning, manipulation, and exploratory analysis)
- NumPy (for numerical operations and array handling)
- Matplotlib (for static visualisations such as histograms, boxplots, and bar charts).
- Seaborn (for enhanced statistical visualisations and correlation analysis)
- Plotly (for interactive visualisations and dashboards)
- SciPy (for statistical testing including t-tests and Mann-Whitney U tests)
- cliffs_delta for efficient effect size calculation with large datasets

**Libraries and Visualisation Approach**
- Matplotlib: Used for histograms, boxplots, and bar charts due to its flexibility in creating custom layouts and side-by-side comparisons. It was particularly effective for visualising log-transformed feature distributions, class balance (count and percentage charts), and distance-based fraud rates.
    - Rationale: Histograms revealed distribution shapes after transformation, while boxplots compared medians between fraud groups, directly supporting the Mann-Whitney U test results.
- Seaborn: Chosen for its statistical styling and ability to produce clean, categorical comparisons. It was used to show fraud rates by online/chip category, repeat retailer status, and purchase price ratio.
    - Rationale: Seaborn’s bar charts provided a visual interpretation of Chi-square test outcomes, turning statistical results into intuitive fraud rate comparisons.
- Plotly: Deployed for interactive visualisations where user exploration adds value, including violin plots (showing distribution shape and outliers for distance from the last transaction) and heatmaps (displaying Cramér’s V associations among categorical variables).
    - Rationale: Plotly’s interactivity allows stakeholders to hover for exact values, zoom into patterns, and explore relationships dynamically. The violin plots and heatmaps also enable deeper inspection of outliers and correlations, consolidating all categorical associations into a single, accessible view.

---
## Project Plan

**Day 1 – Data Understanding and ETL**
- Load, inspect, and clean data in ETL notebook
- Export processed dataset and data dictionary

**Day 2 – Exploratory Data Analysis (EDA)**
- Explore distributions, skew, and correlations
- Use Mann–Whitney U and Chi-square tests
- Identify features for dashboard storytelling

**Day 3 – Modelling and Evaluation**
- Train Logistic Regression and tree-based models 
- Evaluate with Precision, Recall, F1, ROC-AUC, PR-AUC
- Analyse threshold trade-offs

**Day 4 – Dashboard Development (Power BI)**
- Build interactive dashboard with KPIs, filters, and slicers
- Highlight fraud drivers and precision-recall balance

**Day 5 – Review and Documentation**
- Test usability and interpretation
- Finalise README, notebooks, and report

---

## Hypotheses and Statistical Methods

**H1: Fraud increases with greater distance from home**
- Test: Mann-Whitney U test comparing the distributions of distance_from_home for fraudulent and non-fraudulent transactions.
- Rationale: Continuous variable with strong right skew, non-normal distribution, and extreme outliers.

**H2: Fraud increases with greater distance from the previous transaction**
- Test: Mann-Whitney U on distance_from_last_transaction between fraud groups.
- Rationale: Non-parametric test avoids distortion by outliers.

**H3: Fraud increases when the purchase amount is high relative to the customer's usual spend**
- Test: Mann-Whitney U test on ratio_to_median_purchase_price (log-transformed).
- Visualisation: Boxplots and histograms were used to assess skew and highlight outliers.
- Rationale: Extreme values in this feature were expected to correlate strongly with fraud.

**H4: Online orders are more likely to be fraudulent**
- Test: Chi-square test for independence (fraud vs online_order).
- Rationale: Both are categorical variables; this test identifies whether online ordering has a statistically significant association with fraud rate.

**H5: Chip or PIN use reduces the likelihood of fraud**
- Test: Chi-square tests (fraud vs used_chip; fraud vs used_pin_number).
- Rationale: Authentication methods were hypothesised to lower risk. Tests were repeated separately to observe independent and combined effects.

**H6: Fraud likelihood differs between repeat and new retailers**
- Test: Chi-square (fraud vs repeat_retailer).
- Rationale: A measure of familiarity; new retailers were expected to have a slightly higher risk.

**H7: Threshold tuning demonstrates trade-offs between false positives and false negatives**
- Test: Precision-Recall curve analysis and threshold adjustment in the modelling notebook.
- Rationale: This is not a statistical hypothesis - a model evaluation analysis was used to illustrate business trade-offs.

**H8: Fraud risk depends on both channel and chip use**
- Test: Chi-square on the combined categorical variable online_chip_category, created by combining online_order and used_chip.
- Rationale: Tests for interaction between channel type and authentication method.

---
## Project File and Folder Structure

**Folder/File Structure Content Information**

| File / Folder                                  | Purpose / Description                                                                     |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Dashboard/Credit-Card-Analysis-Dashboard.pbix  | Power BI dashboard visualising model insights, KPIs, and fraud detection trade-offs.      |
| data/raw/card_transdata_sample_100k.csv        | Raw stratified sample of transactions used to build the project (source sample).          |
| data/raw/sample_log.json                       | Sampling / ETL log with sampling parameters.                                              |
| data/processed/card_transdata_processed.csv    | Cleaned and feature-engineered dataset ready for modelling and analysis.                  |
| jupyter_notebooks/01_ETL.ipynb                 | ETL: data loading, cleaning, and generation of the processed CSV.                         |
| jupyter_notebooks/02_EDA.ipynb                 | Exploratory data analysis and visualisations; hypothesis checks and summary plots.        |
| jupyter_notebooks/03_Modelling.ipynb           | Modelling: training, validation, threshold analysis, and exporting reports/metrics.       |
| reports/balance_model_comparison.csv           | High-level validation comparison of imbalance-handling strategies (PR AUC / ROC AUC).     |
| reports/balance_threshold_comparison.csv       | Threshold comparison table (Recall / Precision / TP/FP/FN/TN at selected thresholds).     |
| reports/logit_test_thresholds.csv              | Logistic Regression test-set threshold summary (selected thresholds).                     |
| reports/logit_test_thresholds_detailed.csv     | Fine-grained logistic threshold sweep (0.30–0.90) with F1 scores for threshold selection. |
| reports/tree_test_thresholds.csv               | Decision Tree threshold results on the test set.                                          |
| reports/test_model_compare.csv                 | Final model comparison on test set (PR_AUC_Test, ROC_AUC_Test, PR_AUC_Val, Δ).            |
| Wireframe/Balsamiq_Wireframe_Fraud_Overview_Report1.jpg | Initial wireframe design for dashboard layout (Decision Maker view).                      |
| Wireframe/Balsamiq_Wireframe_Threshold_Setting_Report2.jpg | Wireframe design for threshold analysis dashboard (Analyst view).                         |
| README.md                                      | Project documentation and summary of results.                                             |
| requirements.txt                               | Python dependencies and version references for reproducibility.                           |



**File Structure**

```text
Credit_Card-Fraud_Analysis_Updated
├── .git/
├── .gitignore
├── .python-version
├── .venv/
├── Dashboard/
│   └── Credit-Card-Analysis-Dashboard.pbix
├── data/
│   ├── processed/
│   │   └── card_transdata_processed.csv
│   └── raw/
│       ├── card_transdata_sample_100k.csv
│       └── sample_log.json
├── jupyter_notebooks/
│   ├── 01_ETL.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_Modelling.ipynb
├── README.md
├── reports/
│   ├── balance_model_comparison.csv
│   ├── balance_threshold_comparison.csv
│   ├── logit_test_thresholds.csv
│   ├── logit_test_thresholds_detailed.csv
│   ├── test_model_compare.csv
│   └── tree_test_thresholds.csv
├── Wireframe/
│   ├── Balsamiq_Wireframe_Fraud_Overview_Report1.jpg
│   ├── Balsamiq_Wireframe_Threshold_Setting_Report2.jpg
└── requirements.txt
```
---

## Key Findings from EDA

The statistical results confirmed many of the initial hypotheses and provided quantitative support for later modelling decisions.

- H1: Fraudulent transactions occurred at significantly greater distances from home (p < 0.001, Cliff's δ = 0.199, small effect).
- H2: Greater distance from the previous transaction was associated with slightly higher fraud risk (p < 0.001, δ = 0.090, negligible but consistent).
- H3: Higher ratios of purchase price to median spend were strongly associated with fraud (p < 0.001, δ = 0.700, large effect).
- H4: Online transactions had a significantly higher fraud rate (p < 0.001).
- H5: Both chip and PIN use reduced fraud risk, with PIN providing the strongest protective effect (p < 0.001).
- H6: New retailers had a small but statistically significant increase in fraud likelihood (p < 0.05).
- H8: The combination of online and no chip produced the highest observed fraud rate (p < 0.001).

## Modelling

The modelling notebook builds and evaluates fraud detection models using the processed dataset from ETL. It addresses the class imbalance problem (8.74% fraud) and compares different approaches to identify the most effective model for deployment.

**Modelling Libraries and Tools Used**

| Library / Module                       | Purpose / Use in Project                                                                                      |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `sklearn.model_selection`              | Splitting data (train/test/validation), parameter tuning (`train_test_split`, `GridSearchCV`)                 |
| `sklearn.metrics`                      | Calculating evaluation metrics (precision, recall, F1, ROC-AUC, PR-AUC), confusion matrix visualisation       |
| `sklearn.linear_model.LogisticRegression` | Baseline interpretable classifier for comparison                                                            |
| `sklearn.tree.DecisionTreeClassifier`  | Simple, transparent model to explore decision boundaries and feature splits                                   |
| `sklearn.ensemble.RandomForestClassifier` | Ensemble model for non-linear relationships and predictive stability                                        |
| `sklearn.ensemble.GradientBoostingClassifier` | Gradient boosting ensemble for accuracy and bias-variance control                                      |
| `xgboost.XGBClassifier`                | High-performance gradient boosting for advanced modelling                                                      |
| `imblearn.over_sampling.SMOTE`         | Synthetic Minority Oversampling Technique for handling class imbalance                                        |
| `matplotlib.pyplot`                    | Plotting evaluation curves (Precision–Recall, ROC, threshold visualisations)                                  |

### Key steps:

**1. Data Preparation**
- Splits data into train/validation/test sets (60%/20%/20%)
- Isolates test set first to prevent data leakage
- Prepares features (X) and target variable (y) for modelling

**2. Handle Class Imbalance**
- Tests two approaches (SMOTE and class weighting) and includes a control to observe impact.:
    - Class weighting (class_weight='balanced'): Adjusts model to penalise misclassifications of minority class more heavily
    - SMOTE (Synthetic Minority Over-sampling Technique): Generates synthetic fraud examples to balance training data
    - A control (no imbalance handling) was included to determine the impact of using imbalance handling
- **Approach Reasoning:**
    - With only 8.74% fraud cases, models would otherwise learn to predict "non-fraud" for everything

**3. Model Selection**
- Logistic Regression (Balanced): Interpretable baseline, good for understanding feature importance
- Decision Tree (Balanced): Captures non-linear patterns, provides feature importance rankings
- Why these models: Both are interpretable, suitable for imbalanced data, and provide explainable results for business stakeholders

**4. Evaluation Strategy**
- Use PR-AUC (Precision-Recall Area Under Curve) as primary metric:
    - ROC-AUC can be misleading on imbalanced data; PR-AUC is usually more informative because it focuses on the positive class. Nevertheless, ROC AUC reports are included for completeness.

**5. Threshold Analysis**
- Given stakeholders must balance catching fraud with handling false alerts, multiple threshold settings were explored to show precision/recall trade-offs. Tests generated confusion matrices and reports at key thresholds to aid clarity, provide key insights and document findings.

**Note:**
- Based on how well the final models performed (alongside time constraints), pipeline tests (such as pipeline with StandardScaler for Logistic Regression), were not included in the modelling phase.

**Key Takeaways from modelling**
1.  Class imbalance handling (balanced weights or SMOTE) are effective imbalance handling techniques.
2.  Threshold tuning supports determining optimal business thresholds
3.  PR-AUC is more informative than ROC-AUC for imbalanced data
4.  Interpretable models (Logistic Regression, Decision Tree) provide useful actionable insights for stakeholders
5.  Fully synthetic data samples should be used with caution for training (structure can be oversimplified and lack the complexity and depth of real world data for training models).

---

## Dashboard Design

The Power BI dashboard was developed to communicate key findings. It consists of two interactive pages: one providing an overview for managers, and another supporting exploration of threshold setting impacts on the balance between catching fraud and managing false alerts.

The design process began with basic wireframes created in Balsamiq, which are included in the project’s Wireframe folder. These sketches were used to plan the layout of the dashboard design that would support user exploration of the relationship between key fraud drivers.

Each visual is selected to communicate insights derived from analysis of the data, and to demonstrate the implications of business decisions on threshold setting. Filters are used to support interactive exploration of threshold setting, and transaction category or risk factors.

**Dashboard Design Challenges**

The dataset lacked time series and cost data. This limited my ability to show key business insights such as the financial impact of false alerts or missed frauds over time.

**Fraud Overview (Decision Maker View)**
- Purpose: To give a clear, high-level understanding of overall fraud activity, and the operational trade-off between detecting fraud and minimising false alerts.

**Key elements:**
- KPI summary cards displaying:
    - Total number of transactions
    - Total number of incorrect fraud alerts
    - Total correct fraud alerts (true positives)
    - Number of false alerts
- Pie chart showing the percentage of all transactions identified as fraudulent.
- Bar charts showing:
    - Fraud rate by online versus offline purchase type, combined with chip use (used vs not used)
    - Fraud rate by purchase price range (binned ratio_to_median_purchase_price)
- Heatmap showing the relationship between distance from home, purchase price range, and fraud rate.
- Filters and slicers enabling users to interact with dataset features and their influence on fraud detection such as:
    - Online versus in-person transactions
    - Chip and PIN usage
    - Retailer type (new versus repeat retailer)
    - Purchase value and distance from home

This page provides an overview of fraud business impact (quantifying fraud transaction distribution, demonstrating fraud feature patterns, and highlighting the business challenge balance between catching fraud vs managing false alerts).

**Threshold Analysis (Analyst View)**
- Purpose: To allow analysts to explore how adjusting the model's decision threshold impacts performance metrics such as precision, recall, false alerts, and missed frauds.

**Key elements:**
- Threshold slider that dynamically updates the main KPIs:
    - Precision
    - Number of false alerts
    - Number of missed frauds
- KPI cards displaying totals for all transactions and all frauds for reference.
- Bar charts displaying:
    - Fraud rate by PIN use
    - Fraud rate by online versus in-person transactions
    - Fraud rate by chip use category (online, offline, with and without chip)
    - Fraud rate by distance from home
    - Fraud rate by distance from last transaction
    - Fraud rate by purchase price band
- Filters to enable exploration of fraud features (including some combined features such as chip/PIN and offline/online).
- The goal was to bridge the gap between technical threshold analysis and practical business application.

---

## Ethical Considerations and Responsible AI

Although this project uses a synthetic dataset with no identifiable personal information, in a real-world context, fraud detection involves sensitive personal and financial data. All personal data processing must comply with the General Data Protection Regulation (GDPR) and the Data Protection Act 2018.

**Accountability and Human Oversight**

Fraud detection systems should not operate without human oversight. Whilst automation plays a vital role in catching and preventing fraud, human review also plays a critical role in verifying alerts and maintaining accountability.

A human-in-the-loop approach ensures that model predictions are not left unchecked. For example, a flagged transaction might be manually verified through customer contact before any account restrictions are imposed. This ensures genuine transactions incorrectly flagged are identified. Research indicates that customer churn can be linked to the impact of incorrectly blocked genuine transactions.

---

## Project Challenges and Limitations

**Technical Challenges**

Several practical challenges arose during the development and implementation stages of this project:

**Environment setup and dependencies**
- At the start of the project, Visual Studio Code and kernel issues initially caused delays. Once resolved, the environment remained stable until VS Code announced virtual environment issues. This caused a secondary delay to getting work completed.

**Seaborn compatibility**
- An update to Seaborn deprecated the use of colour palettes without a defined hue, triggering repetitive FutureWarning messages. This was corrected by explicitly adding the parameters hue= and legend=False to affected plots. This issue was identified and resolved using a combination of GitHub Copilot suggestions and manual testing.

**Cliff's Delta output**
- When applying the cliffs_delta function, the initial implementation produced a TypeError because the function returned a tuple rather than a single float value. Making a code change to extract the first element of the tuple resolved the issue.

**Model convergence**
- The Logistic Regression model (non-weighted) triggered convergence warnings at 2000, reducing this value produced stable and reproducible results without altering accuracy or coefficients. On later tests even at 2000 the convergence warnings disappeared, this is a factor I would have liked to investigate if time had permitted.

**Dataset and Modelling Limitations**
- The synthetic nature of the dataset introduced some constraints when assessing real-world applicability:
    - The clear separation between classes led to unusually high model performance scores, particularly for tree-based methods, which are well-suited to rule-like relationships. This did not reflect the complexity of real fraud data, where genuine and fraudulent behaviour overlap extensively.
     - The lack of time and cost-related features prevented key analysis of financial or time-based impacts. Without transaction amounts (or time series information), it was impossible to estimate total fraud loss or time-based risk factors.
    - The synthetic nature of the dataset reduced correlation complexity. Training and test performance remained very consistent, such near-perfect results pointed to overly clear feature separation. This factor limited exploration of more advanced modelling techniques.

---

## Project Summary

This project explored how data analysis and machine learning can support fraud detection while balancing operational risk, false alerts, and investigation cost. The aim was not only to detect fraudulent transactions effectively but to understand how model thresholds and decision settings influence business outcomes.

It explores how financial institutions must balance the cost of missed fraud (false negatives) against the cost of investigating false alerts (false positives).

Through the use of a structured workflow (covering ETL, EDA, modelling, and dashboard design) the project demonstrated how analytical findings can be transformed into clear, actionable insights for both technical and non-technical audiences.

Insights drawn were that distance from home, purchase price ratio, and online transactions were the strongest indicators of fraud, while authentication features such as chip and PIN use significantly reduced risk. Modelling provided additional clear insights into feature importance and highlighted the simplicity of fraud separation within the synthetic dataset.

## Credits

**Fraud Research and Business Context**

- FICO (2023). *Credit Card Fraud Detection: 2025 Trends and Interventions.*
- Soto-Valero, C. (2023). *Evaluation Metrics for Real-Time Financial Fraud Detection ML Models.*
- Alon, T. (2023). *Ultimate Guide to PR-AUC: Calculations, Uses, and Limitations.* Coralogix Blog.
- Sahoo et al. (2023). *Data Leakage and Deceptive Performance in Credit Card Fraud Detection.* arXiv preprint.
- NICE Actimize (2024). *Reducing False Positives in Transaction Monitoring.*
- J.P. Morgan (n.d.). *False Positives & Fraud Prevention Tools.*

**Workflow and problem resolution**

- Tutor: Vasi Pavaloi, for guidance on code troubleshooting and support throughout the project.
- ChatGPT and GitHub Copilot, used for code debugging and clarification during development.
- Code Institute LMS documentation, which provided examples of structure and best practice for technical documentation.
