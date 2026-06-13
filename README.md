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
├── bank_transactions.csv          # Raw client data (1M+ Records)
├── bank_segmentation_pipeline.py  # Production Python script
├── rfm_scaler.pkl                 # Saved StandardScaler parameters (μ, σ)
├── rfm_kmeans_model.pkl           # Trained final K-Means model artifact
├── Customer_Segmentation.pbix     # Interactive Power BI Dashboard
├── bank_customer_segments.csv     # Final output with Cluster & Segment IDs
└── README.md                      # Project Documentation
