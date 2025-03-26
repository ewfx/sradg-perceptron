# 🚀 Project Name

## 📌 Table of Contents

-   [Introduction](#introduction)
-   [Demo](#demo)
-   [Inspiration](#inspiration)
-   [What It Does](#what-it-does)
-   [How We Built It](#how-we-built-it)
-   [Challenges We Faced](#challenges-we-faced)
-   [How to Run](#how-to-run)
-   [Tech Stack](#tech-stack)
-   [Team](#team)

---

## 🎯 Introduction

A modular AI-powered reconciliation system that handles bank statement vs. book record matching with dynamic column detection, multi-agent analysis, and corrective actions.

## 🎥 Demo

🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## ⚙️ What It Does

Automates financial reconciliation using AI agents to detect mismatches (amounts, dates, fraud), suggest fixes, and escalate issues. Handles dynamic report formats via LLM-powered column detection and offers self-healing corrections.

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

-   **Precision-Recall Tradeoff**

    -   **Issue**: Over-aggressive auto-matching caused false negatives.
    -   **Fix**: Confidence scoring + human-in-the-loop validation.

-   **Currency Flux**

    -   **Hurdle**: Real-time vs historical forex rate mismatches.
    -   **Resolution**: Blend ECB API rates + internal historic averages.

-   **Audit Trail Integrity**

    -   **Risk**: Immutable logging of automated changes.
    -   **Implementation**: Blockchain-style hashing for critical operations.

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
    conda env create -f environment.yml  # or pip install -r requirements.txt (for Python)
    ```
3. Run the project
    ```sh
    python code/src/myagent.py  # or python app.py
    ```

## 🏗️ Tech Stack

-   🔹 Backend: Fast API
-   🔹 Database: PostgreSQL, Vector Store
-   🔹 Other: OpenAI API, GroqCloud
