# 🚀 Project Name

## 📌 Table of Contents

-   [Introduction](#introduction)
-   [Demo](#demo)
-   [What It Does](#what-it-does)
-   [How We Built It](#how-we-built-it)
-   [Challenges We Faced](#challenges-we-faced)
-   [How to Run](#how-to-run)
-   [Tech Stack](#tech-stack)

---

## 🎯 Introduction

🚀 Overview
This repository contains a fully agentic AI reconciliation system that:

-   ✅ Detects and resolves anomalies in financial transactions
-   ✅ Uses LLM-driven decision-making to automate reconciliation
-   ✅ Performs tool-based investigations (query history, update records, escalate issues, etc.)
-   ✅ Minimizes false positives using statistical thresholds & ML-based anomaly detection
-   ✅ Supports auditability through structured tool execution logs

This AI agent mimics real-world Ops teams, handling missing transactions, amount mismatches, fraud detection, and resolution autonomously.

## 🎥 Demo

📹 [Video Demo](https://drive.google.com/file/d/1i5L-dQFK9_K0KpVqSLHgTEs1G3c6mBpy/view?usp=sharing)  
🖼️ Screenshots:

![Screenshot 1](artifacts/pictures/ss1.png)
![Screenshot 2](artifacts/pictures/ss2.png)
![Screenshot 3](artifacts/pictures/ss3.png)

## ⚙️ What It Does

-   **AI-Driven Anomaly Detection**

    -   Compares transactions from Bank & Ledger to identify mismatches.
    -   Applies machine learning (Isolation Forest, Statistical Analysis) to detect unusual transactions.
    -   Uses FAISS (or other historical search methods) to find similar past anomalies.

-   **Fully Autonomous Decision Making (LLM-Powered)**

    -   Uses GPT-based LLM to determine the best resolution for each anomaly.
    -   Executes corrective actions based on predefined thresholds.
    -   Calls investigation tools to collect more details when required.

-   **Dynamic Tool Execution for Resolution**

    -   Historical Data Query: Find similar past anomalies.
    -   Statistical Analysis: Detect amount & timing variances.
    -   Transaction Addition & Removal: Adjust incorrect records.
    -   Currency Conversion: Fix discrepancies due to exchange rates.
    -   Alerts & Escalations: Notify finance teams for manual review.
    -   Prediction Tool: Suggest next steps based on past resolutions.

-   **Explainable & Auditable Actions**
    -   Every tool execution logs structured requests & responses in XML format.
    -   Confidence scores help auditors understand how decisions are made.
    -   Generates plain English summaries for financial reports.

## 🛠️ How We Built It

-   **Core**: Python, Pandas, Scikit-learn, Transformers, HuggingFace

-   **AI**: DeepSeek R1, OpenAI (column detection), GroqCloud, Isolation Forest (anomalies), SQLite (querying)

-   **Agents**: 12 specialized tools for analysis/corrections

-   **Infra**: In-memory DB for speed, Preprocessing pipeline for normalization

## 🚧 Challenges We Faced

-   **Reconciliation Process**

    -   **Problem**: Understanding the process of reconciliation, all the actions one takes and comes to a conclusion.
    -   **Solution**: Office Hours + YouTube videos + Self Analysis.

-   **Dynamic Schema Handling**

    -   **Problem**: Varying column names/date formats across reports.
    -   **Solution**: Hybrid LLM + fuzzy matching with fallback rules.

-   **Agent Orchestration**
    -   **Complexity**: Avoiding circular tool dependencies.
    -   **Design**: State machine with short-lived contexts.

## 🏃 How to Run

1. Clone the repository
    ```sh
    git clone https://github.com/ewfx/sradg-perceptron.git
    ```
2. Install dependencies

    ```sh
    conda env create -f environment.yml
    conda activate ghack
    ```

3. Create a .env file and add following keys

    ```
    GROQ_API_KEY=
    HUGGINGFACEHUB_API_TOKEN=
    ```

4. Run the project
    ```sh
    python code/src/myagent.py  # or python app.py
    ```

## 🏗️ Tech Stack

-   🔹 Backend: Fast API
-   🔹 Database: PostgreSQL, Vector Store
-   🔹 Other: OpenAI API, GroqCloud
