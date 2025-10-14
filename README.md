# UK Sepsis Early Detection Model: A NEWS2-Based ML Approach

## 1. Project Overview and Clinical Context
This project presents a foundational Machine Learning model designed to predict the early onset of sepsis using key physiological parameters derived from the UK's **National Early Warning Score (NEWS2)** system.

The core objective is to demonstrate the feasibility of an automated, low-latency screening tool that can flag high-risk patients based on readily available vital signs, supporting timely clinical intervention.

## 2. Technical Implementation
This proof-of-concept utilizes a simple yet effective classification pipeline built in Python.

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Data Handling** | `pandas` | Loading, cleaning, and structuring the mock vital signs data. |
| **Model** | `sklearn.linear_model.LogisticRegression` | The core algorithm used to learn the relationship between vitals and sepsis outcome. |
| **Evaluation** | `sklearn.metrics` | Used to calculate the model's performance metrics. |

### Data Features
The model's inputs (`X`) are based on five key NEWS2-relevant vital signs, plus Level of Consciousness (`LOC_Alert`). The target (`y`) is a binary outcome (1 = Sepsis Onset, 0 = No Sepsis).

## 3. Results and Performance
The model was trained and tested on a small, synthetic dataset to validate the pipeline.

**Final Accuracy Score: 100.00%**

This perfect score, achieved on the test set (2 data samples), confirms that the Logistic Regression model successfully identified the distinct patterns separating sepsis cases from non-sepsis cases in the mock data.

Classification Report (Excerpt)

Metric	Class 0 (No Sepsis)	Class 1 (Sepsis)
Precision	1.00	1.00
Recall	1.00	1.00
F1-Score	1.00	1.00

## 4. How to Run the Project
To reproduce these results, execute the following commands in your project's terminal:

A. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

B. Execute the Model

```bash
python3 model_script.py
```

## 5. Next Steps

Future iterations of this work should focus on integrating this model with a much larger, anonymized clinical dataset to validate its robustness and generalizability in a real-world setting.

### ⚠️ DISCLAIMER: Educational Use Only

THIS IS NOT A CLINICAL DIAGNOSTIC TOOL. This project is developed purely for educational and portfolio purposes using simulated data. 
