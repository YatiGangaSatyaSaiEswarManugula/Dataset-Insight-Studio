# Dataset Insight Studio

An adaptive analytics studio that can upload and analyze different CSV datasets. It combines Python analysis, forecasting, Streamlit dashboards, a sample dataset, and optional local LLM support through Ollama.

## Features

- Streamlit web app for CSV upload, dataset profiling, charts, and query-driven analysis
- Dataset forecasting with monthly aggregation and a `RandomForestRegressor`
- Natural-language query handling for forecasts, top performers, recommendations, and sales goals
- Generic CSV support for basic profiling and forecasting when a dataset has date and metric columns
- Optional local LLM answers using Ollama, without a paid API key
- Exportable CSV outputs, charts, and a project report

## Project Files

- `streamlit_app.py` - Streamlit dashboard and interactive query interface
- `train_amazon_forecast.py` - forecasting, rule-based analysis, and command-line workflow
- `local_llm_assistant.py` - Ollama availability checks and local LLM prompt handling
- `Amazon.csv` - base dataset used by default
- `PROJECT_REPORT.md` - full project write-up
- `ARCHITECTURE_DIAGRAM.md` - Mermaid architecture diagram

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app uses `Amazon.csv` by default. You can upload another CSV from the Streamlit sidebar to analyze a different dataset.

## Command-Line Example

```powershell
python train_amazon_forecast.py --query "predict electronics sales in India for next 6 months"
```

## Optional Local LLM Mode

Install and run Ollama if you want AI-style answers from a local model. The app works in rule-based mode even when Ollama is not running.

