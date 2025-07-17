# 🏦 Customer Churn Analysis for Retail Banking

## **Business Question**
What are the most significant factors leading to customer churn (attrition) for a retail bank, and what actionable insights can be derived from exploratory data analysis to guide retention strategies?

---

## 1. 📌 Business Overview and Importance

**Customer churn** leads to revenue loss and higher customer acquisition costs. Understanding churn helps banks:
- Identify at-risk segments
- Develop targeted retention strategies
- Improve customer lifetime value

Retention is more cost-effective than acquisition, making churn reduction vital for sustainable growth.

---

## 2. 🗂 Data Overview

- **Source**: Kaggle – Bank Customer Churn Prediction  
- **Observations**: 10,000 customers with 14 variables  
- **Categories**:
  - *Demographics*: Geography, Gender, Age
  - *Account Features*: CreditScore, Tenure, Balance
  - *Behavioral*: NumOfProducts, IsActiveMember
  - *Target*: Exited (1 = churned)

**Removed Columns**:
- RowNumber, CustomerId, Surname (for privacy and model focus)

> **Business Implication**: The dataset supports holistic, data-driven decision-making for segmentation and retention.

---

## 3. 🔧 Data Preprocessing: Ensuring Relevance and Quality

### Key Steps:
- **Missing Values**: Dropped 9 rows with missing CreditScore → final shape: `9,991 × 14`
- **Duplicates**: Removed to prevent analytical bias
- **Validation**: Checked for negative or invalid feature values
- **Irrelevant Features**: Dropped identifiers for GDPR compliance

> **Business Implication**: Clean, consistent data ensures trustworthy insights and regulatory alignment.

---

## 4. 📊 Exploratory Data Analysis (EDA)

### 4.1 Univariate Analysis

#### Key Insights:
- **Credit Score**:  
  - Mean = 650.5 | Std Dev = 96.7  
  - High variability suggests segment-based offers
- **Geography & Gender**:  
  - France dominant; 54.6% male  
  - Supports cultural or gender-specific targeting
- **Age**:  
  - Mean = 38.9 (younger-skewed)  
  - Tailored retention by life stage may help
- **Balance**:  
  - Avg = 76,485; many with **zero** balance  
  - Zero balance clients may be at churn risk
- **Num of Products**:  
  - Most customers have 1–2 products  
  - Opportunity for cross-selling
- **Credit Card & Activity**:  
  - 70.5% have credit cards; only 51.5% are active  
  - Engagement strategies needed
- **Estimated Salary**:  
  - Wide income range = diverse financial profiles
- **Churn Rate**:  
  - **20.4%** — high and actionable

### Visuals:
- Histograms and box plots showed:
  - High zero-balance frequency
  - Broad distribution for salary and balance

---

### 4.2 Bivariate Analysis

#### Correlation Matrix:

| Feature Pair              | Correlation | Insight                                             |
|---------------------------|-------------|------------------------------------------------------|
| Age vs. Exited            | 0.29        | Older customers more likely to churn                |
| Balance vs. Exited        | 0.12        | Weak positive relationship                          |
| NumOfProducts vs. Balance | -0.30       | Fewer products → higher balances                    |
| CreditScore & Salary      | ≈ 0         | Minimal predictive power for churn                  |

> Scatter and pair plots show churn clusters in older clients with high or zero balances and fewer products.

---

### 4.3 Multivariate Analysis

- **Heatmaps / Pair Plots**:
  - Age + Balance = dominant churn predictors
  - Multiple products → lower churn likelihood
- **PCA**:
  - PC1 & PC2 explain **38.75%** variance
  - Churners (orange clusters) show pattern but can't be linearly separated

> **Business Takeaway**: Advanced models are needed beyond linear trends to predict churn.

---

## 5. 💡 Strategic Insights and Recommendations

- **Retention Focus Areas**:
  - Older customers → loyalty benefits, personalized communication
  - Customers with zero or very high balances → proactive engagement
  - Encourage product diversification through cross-sell/upsell offers

- **Deprioritize**:
  - Credit Score & Estimated Salary — low impact on churn prediction

- **Geographic Segmentation**:
  - Customize offers or service delivery by region

---

## 6. ⚠️ Limitations & Next Steps

### Limitations:
- Correlation ≠ causation
- No time-series or behavior trends over time
- No qualitative metrics (e.g., satisfaction, complaints)

### Next Steps:
- Implement machine learning churn models
- Test interventions for at-risk segments
- Incorporate satisfaction and survey data for richer context

---

## 7. ✅ Conclusion

This analysis provides a foundation for targeted, data-driven customer retention. By focusing on churn-related factors such as:
- Age
- Balance
- Product usage
- Engagement

…the bank can build personalized outreach and reduce its 20.4% churn rate. Continued investment in predictive modeling and customer feedback will refine strategies and improve customer lifetime value.

---

📎 **Full Report & Visuals**:  
Read thge Report.pdf file
