# 📌 Final Project: Machine Learning  

## 📅 Date: [04/05/2025]  
**Author:** Lucía Herraiz Cano  

---  

## 📖 Introduction  

This repository contains two predictive models designed to estimate students' grades in two educational institutions, considering a series of determining factors and analyzing which ones are most relevant.  

---  

## 📂 Contents  
1. [🔹] [General Description](#general-description)  
2. [⚙️] [Installation and Execution](#installation-and-execution)  
3. [📊] [Module Distribution] (#module-distribution)
4. [🚀] [Results and Conclusions](#results-and-conclusions)  
5. [📜] [References](#references)  

---  

## 🔹 General Description  

This project develops predictive models to analyze the factors influencing students' academic performance in two high schools in Madrid in 2005. Using demographic, social, and academic data, two models are built: one that includes previous grades and another that excludes these values to assess the influence of other factors. The objective is to understand which variables have the most impact on the final grade and explore strategies to improve student performance.  

---  

## ⚙️ Installation and Execution  

### 🚀 Prerequisites  

**Compatible operating system:** Windows, macOS, or Linux.  

**Required software:**  
- Visual Studio Code (optional but recommended).  
- Git installed and configured.  
- Python.  

**Specific dependencies:** Check the *requirements.txt* file to install the necessary packages.  

**Access paths:** Access paths and other parameters can be configured in the *configuration.ini* file.  

### 🔧 Installation Steps  

```bash
# Clone the repository
git clone https://github.com/LuciaHC/Grade_Prediction_Models.git  

# Navigate to the src folder 
cd src/  

# Install dependencies
pip install -r requirements.txt  

# Run the environment
python main.py  

```
---

## 📖 Module Distribution

The project is divided in the following modules (inside \src folder):

- Models: Manual implementation of 9 regression or classification models
- notebooks
    - *Exploratory_Data_Analysis*: Initial exploration of data
    - *Model_Testing*: Algorithms and strategies tried for Model 1 and Model2
    - *Final_Model_Tuning*: Cross-validation process for the best performing algorithms
    - *Metrics_Evaluator*: Achievement of reliable final metrics 
- utils: Functions used for evaluating and plotting the models
- data_processing: Functions related to data cleaning and processing

---

## 📊 Results and Conclusions

All of the results and the conclusions of this proyect (in spanish) can be consulted in the document *Informe.pdf*


---
## 📜 References

