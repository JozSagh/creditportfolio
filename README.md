<h1 align="center">Credit Risk PD Model</h1>
<h3 align="center">Production-Oriented Probability of Default Framework</h3>

<hr>

<h2>📌 Overview</h2>

<p>
This repository demonstrates a <strong>real-life style Probability of Default (PD) modeling framework</strong> 
built using machine learning techniques while taking into account regulatory and credit risk best practices.
</p>

<p>
The project simulates a real-world retail credit risk scenario and implements an 
end-to-end modeling pipeline — from data generation to calibrated score transformation.
</p>

<p>
The objective is not only predictive performance, but <strong>robust, regulator-ready modeling architecture</strong>.
</p>

<hr>

<h2>🏗 Key Modeling Components</h2>

<ul>
  <li><strong>Customer-Level Data Leakage Prevention</strong>  
      <br>GroupKFold ensures that customers never appear in both training and validation sets.</li>
  
  <li><strong>Out-of-Time (OOT) Validation</strong>  
      <br>Model trained on January–November data and validated on December contracts to simulate forward-looking performance.</li>
  
  <li><strong>XGBoost-Based PD Modeling</strong>  
      <br>Nonlinear gradient boosting classifier suitable for financial risk modeling.</li>
  
  <li><strong>Model Explainability (XAI)</strong>  
      <br>SHAP values used for global and local interpretability.</li>
  
  <li><strong>PD Calibration</strong>  
      <br>Platt Scaling applied to align predicted probabilities with observed default rates.</li>
  
  <li><strong>10-Bin Risk Segmentation</strong>  
      <br>Decile-based analysis for portfolio monitoring and cut-off strategy design.</li>
  
  <li><strong>Credit Score Transformation</strong>  
      <br>PD converted into score using PDO methodology (Points to Double the Odds).</li>
</ul>

<hr>

<h2>📊 Methodological Highlights</h2>

<table>
  <tr>
    <th>Component</th>
    <th>Implementation</th>
  </tr>
  <tr>
    <td>Cross-Validation</td>
    <td>GroupKFold (customer-level grouping)</td>
  </tr>
  <tr>
    <td>Temporal Validation</td>
    <td>Out-of-Time (OOT) December Holdout</td>
  </tr>
  <tr>
    <td>Model</td>
    <td>XGBoost Classifier</td>
  </tr>
  <tr>
    <td>Explainability</td>
    <td>SHAP (TreeExplainer)</td>
  </tr>
  <tr>
    <td>Calibration</td>
    <td>Sigmoid (Platt Scaling)</td>
  </tr>
  <tr>
    <td>Segmentation</td>
    <td>10 Quantile Risk Bins</td>
  </tr>
  <tr>
    <td>Scorecard Logic</td>
    <td>PDO-based log-odds transformation</td>
  </tr>
</table>

<hr>

<h2>🧠 Synthetic Data Design</h2>

<p>
Synthetic dataset includes:
</p>

<ul>
  <li><strong>customer_id</strong> (grouped observations)</li>
  <li><strong>month</strong> (1–12)</li>
  <li>income</li>
  <li>age</li>
  <li>existing_debt</li>
  <li>utilization_rate</li>
  <li>target (default flag)</li>
</ul>

<p>
Default probability is generated using a structured logistic function to simulate realistic economic behavior.
No real customer data is used.
</p>

<hr>

<h2>🚀 How to Run</h2>

<pre>
pip install -r requirements.txt
python main.py
</pre>

<p>
Pipeline outputs:
</p>

<ul>
  <li>Cross-validated AUC</li>
  <li>Out-of-Time AUC</li>
  <li>Confusion matrix</li>
  <li>ROC curve</li>
  <li>SHAP feature importance</li>
  <li>Calibration curve</li>
  <li>Decile analysis</li>
  <li>Generated credit scores</li>
</ul>

<hr>

<h2>🎯 What This Project Demonstrates</h2>

<ul>
  <li>Proper grouped cross-validation design</li>
  <li>Temporal stability validation</li>
  <li>Integration of ML with traditional credit risk methodology</li>
  <li>Probability calibration best practices</li>
  <li>Scorecard-style transformation logic</li>
  <li>Explainable AI in financial risk modeling</li>
</ul>
