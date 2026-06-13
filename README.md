# Bank Customer Segmentation Pipeline (Unsupervised ML)

An end-to-end, production-grade machine learning pipeline that clusters over 1 million retail banking customer records using an optimized RFM framework and K-Means clustering.

## 🚀 Project Overview
This project transforms raw transactional data into actionable customer segments to drive personalized banking strategies. It features automatic outlier treatment, variance stabilization, mathematical cluster optimization, and model serialization for deployment.

### Tech Stack
* **Language:** Python 3 interpreter
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
* **Business Intelligence:** Power BI Desktop
* **Deployment Assets:** Pickle (`.pkl`) Serialization

---

## 📦 Repository Structure
```text
📁 bank-customer-segmentation-ml/
│
├── 📄 README.md                        <-- The matrix table here!
│
├── 📁 notebooks/
│   └── 📓 01_Data_Audit.ipynb          <--  Interactive exploratory notebook
│
├── 📁 src/
│   ├── 🐍 01_data_audit.py             <--  Cleaning script artifact
│   └── 🐍 bank_segmentation_pipeline.py <--  Production ML pipeline script
│
├── 📁 models/
│   ├── 📦 rfm_scaler.pkl               <-- Serialized StandardScaler constants
│   └── 📦 rfm_kmeans_model.pkl         <-- Serialized trained KMeans model brain
│
└── 📁 dashboard/
    ├── 📊 Customer_Segmentation.pbix   <-- Interactive Power BI file
    └── 📄 bank_customer_segments_final.csv <-- Labeled dataset powering the BI visuals
```
---

## 📊 End-to-End Machine Learning Pipeline

### Phase 1 & 2: Data Loading, Audit & Cleaning
- Listwise Deletion: Applied structural validation across the 1.04M+ record base. Rows with null entries in critical features (CustAccountBalance, CustGender, CustLocation) were omitted to preserve absolute data quality without diminishing statistical power.
- Outlier Treatment: Implemented a 99th-percentile Winsorization boundary on highly skewed transaction amounts to prevent spatial distortion of cluster centroids.

### Phase 3 & 4: RFM Feature Engineering & Variance Stabilization
- Feature Construction: Calculated indicators relative to a centralized operational snapshot boundary:
- Recency: Days elapsed since the consumer’s most recent transaction.
- Frequency: Aggregate transaction volume recorded per unique customer account.
- Monetary: Summation of all capped transaction balances per customer.
- Log Transformation: Applied a natural logarithmic transformation (np.log1p) across all engineered attributes to smooth sharp right-skewed normal profiles and stabilize regional feature variance before distance computation.
- Feature Scaling: Standardized features to zero mean and unit variance using StandardScaler. 

### Phase 5, 6 & 7: Model Optimization & Mathematical Validation
- Hyperparameter Tuning: Iterated K-Means algorithms sequentially from K=2 through K=10 utilizing smart k-means++ initialization centers to remove convergence failures.
- Mathematical Selection: Evaluated clustering metrics to identify an optimal inflection threshold at exactly K=4 via the Elbow Method (WCSS/Inertia minimization curve), validated by a peak Silhouette Score of 0.52.

---
### 💼 Customer Persona Matrix & CRM Action Strategies
Following statistical evaluation, centroids were reverse-mapped to their raw un-transformed averages to identify four distinct commercial archetypes:
| Cluster ID | Segment Persona | Behavioral Fingerprint | CRM Communication Channel | Business Priority |
| :---: | :--- | :--- | :--- | :---: |
| **Cluster 0** | **Champions** | Extremely low recency, high frequency, and maximum financial spending profiles. | Premium Perks, Private Wealth Cards, & Dedicated Account Managers | **High** |
| **Cluster 1** | **Dormant Low-Value** | Protracted inactivity (>1 year) with minimal transaction frequency. | Automated Low-Cost Win-back Emails & Push Notifications | **Medium** |
| **Cluster 2** | **Loyal Mid-Tier** | Consistent routine transactional frequency paired with moderate ticket values. | In-App Popups, Cross-Selling targeted Systematic Investment Plans (SIPs) & Personal Loans | **High** |
| **Cluster 3** | **At-Risk High-Spenders** | High historical financial value but exhibiting severe recent drop-offs in transaction interaction. | Direct Phone Outreach from Branch Managers, Fee Waivers, & Tailored Retention Offers | **Critical** |

---
### 🎯 Production Blueprint & Real-Time Scoring
A primary highlight of this pipeline is its production readiness. Instead of producing a static, one-time analysis, the script serializes pipeline parameters into .pkl objects:
- rfm_scaler.pkl: Stores scaling constants (\mu, \sigma) so that incoming daily customer transactions can be mapped to the exact same transformation scale.
- rfm_kmeans_model.pkl: Stores the mathematical centroid coordinates, allowing the bank to run instant real-time segmentation on incoming data without rebuilding or retraining the full network.

---
BANK CUSTOMER SEGMENTATION PIPELINE 
 
Author : Sakshi Kumari

Program  : B.Tech CSE (Data Science)
  
