# Dataset Insight Studio Project Report

## 1. Project Title
Dataset Insight Studio: A Streamlit-Based Data Analysis, Forecasting, and Local LLM Question-Answering System

## 2. Project Objective
The goal of this project is to build an interactive analytics application that allows a user to:

- upload a dataset
- analyze the dataset automatically
- ask natural-language questions about the dataset
- generate forecasts when time-based data is available
- receive strategy and recommendation-style answers
- work without a paid API by using a local LLM through Ollama

This project started from an Amazon sales analysis workflow and was later extended to support generic CSV datasets.

## 3. Problem Statement
Most beginner analytics projects stop at dashboards or notebooks. They can show charts and tables, but they do not let users interact naturally with the data. The challenge in this project was to create a system that combines:

- data analysis
- business insights
- forecasting
- user-driven queries
- offline, no-API AI support

The result is a hybrid application that uses both rule-based analytics and a local LLM.

## 4. Tools and Technologies
- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- Ollama
- Local LLM models such as `gemma3:4b`

## 5. Project Files
- [streamlit_app.py](streamlit_app.py)
- [train_amazon_forecast.py](train_amazon_forecast.py)
- [local_llm_assistant.py](local_llm_assistant.py)
- [Amazon.csv](Amazon.csv)

## 6. Core Features

### 6.1 Amazon-Specific Rule Engine
For datasets that match the Amazon schema, the system can:

- forecast future sales
- analyze categories, countries, order status, and trends
- answer recommendation queries
- generate goal-based plans such as increasing sales by a target percentage

### 6.2 Generic Dataset Support
For non-Amazon CSV files, the system:

- detects whether the uploaded dataset is generic
- infers numeric, categorical, and date-like columns
- creates a generic dataset summary
- uses a local LLM to answer natural-language questions from the uploaded dataset

### 6.3 Local LLM Integration
The project supports offline AI-style answers using Ollama. This removes the need for a paid API. The app:

- checks whether Ollama is available
- detects installed local models
- sends dataset context plus user query to the model
- uses the LLM for broader question answering

### 6.4 Hybrid Reasoning
The application does not rely only on the LLM. It first tries to compute structured results when possible. These structured outputs are then passed to the local LLM as extra context. This improves answer quality and grounding.

## 7. Workflow

### Step 1: Dataset Input
The user uploads a CSV file through the Streamlit sidebar.

### Step 2: Schema Detection
The app checks whether the uploaded dataset is:

- an Amazon-style dataset
- or a generic CSV dataset

### Step 3: Data Analysis
Depending on the schema, the app:

- performs rule-based analysis for Amazon data
- or generates a generic analysis summary for any other dataset

### Step 4: User Query Handling
The user enters a natural-language query such as:

- `predict electronics sales in India for next 6 months`
- `Which 3 actions will maximize profit?`
- `What should I do to increase sales by 20% next month?`
- `What will be the total sales for the next quarter?`

### Step 5: Answer Generation
The answer is produced by:

- rule-based analytics for structured supported queries
- local LLM responses for free-form queries
- or a hybrid combination of both

## 8. Forecasting Logic
Forecasting is based on monthly aggregated values and uses:

- lag features
- rolling means
- seasonal month encoding
- `RandomForestRegressor`

The model is trained on historical monthly data and predicts future values such as next month, next quarter, or next six months.

## 9. Generic Forecasting Support
For uploaded datasets like Superstore, the app now attempts to:

- identify a date column
- identify a likely sales or revenue column
- aggregate values monthly
- build a generic forecast
- provide that forecast as structured context to the local LLM

This improves answers for generic time-series questions.

## 10. Sample Dataset
The project includes a base dataset:

- [Amazon.csv](Amazon.csv)

This dataset:

- follows the Amazon schema
- contains data from January 2022 to March 2026
- supports forecasting, recommendations, and goal-based planning

## 11. Key Improvements Made During Development
- added Streamlit web interface
- added natural-language query handling
- added recommendation mode
- added goal-planning mode
- added unsupported-query handling
- integrated local LLM support through Ollama
- added generic dataset support
- improved CSV encoding compatibility
- added generic forecasting support for uploaded non-Amazon datasets

## 12. Current Limitations
- the rule-based engine is still specialized for the Amazon dataset
- the quality of free-form answers depends on the local LLM model used
- generic forecasting depends on correctly identifying date and metric columns
- completely arbitrary queries may still require a stronger model or more advanced retrieval logic

## 13. Future Scope
- add conversational memory across multiple user questions
- support Excel files (`.xlsx`)
- add delimiter auto-detection for CSVs
- allow users to manually choose forecast target columns
- add richer generic charts and profiling dashboards
- support more dataset-specific analytical templates

## 14. Conclusion
This project demonstrates how a traditional data analysis application can be extended into an intelligent analytics assistant. It combines:

- structured analytics
- forecasting
- business recommendations
- user query interpretation
- offline local LLM integration

As a result, the project goes beyond static analysis and becomes an interactive decision-support tool.

## 15. How to Run
```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To use local LLM mode, Ollama must be installed and running.

