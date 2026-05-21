# Architecture Diagram

```mermaid
flowchart TD
    A["User"] --> B["Streamlit Web App<br/>streamlit_app.py"]

    B --> C["Dataset Upload / Default CSV"]
    C --> D["Schema Detection"]

    D --> E["Amazon Schema Path"]
    D --> F["Generic Dataset Path"]

    E --> G["Rule-Based Analytics Engine<br/>train_amazon_forecast.py"]
    G --> G1["Forecasting"]
    G --> G2["Analysis"]
    G --> G3["Recommendations"]
    G --> G4["Goal Planning"]

    F --> H["Generic Dataset Analyzer"]
    H --> H1["Date Column Detection"]
    H --> H2["Metric Column Detection"]
    H --> H3["Grouped Forecast Detection"]
    H --> H4["Generic Summary Context"]

    B --> I["Ollama Availability Check"]
    I --> J["Local LLM Mode"]
    I --> K["Rule-Only Mode"]

    J --> L["local_llm_assistant.py"]
    L --> M["Dataset Context Builder"]
    M --> N["Supplemental Structured Context"]
    N --> O["Ollama Local Model"]
    O --> P["Natural Language Answer"]

    K --> Q["Structured UI Output"]

    G --> N
    H --> N

    P --> R["Streamlit Response"]
    Q --> R

    R --> S["Charts"]
    R --> T["Forecast Tables"]
    R --> U["Recommendations / Goal Plans"]
    R --> V["Downloadable Outputs"]
```

## Short Explanation

The application starts when the user interacts with the Streamlit app and either uploads a CSV file or uses the default dataset. The system then detects whether the uploaded file matches the Amazon schema or is a generic dataset.

If the dataset matches the Amazon schema, the rule-based analytics engine handles forecasting, analysis, recommendations, and goal planning. If the dataset is generic, the app performs generic column detection and builds a generalized data context.

When local LLM mode is enabled, the app checks Ollama availability and sends both dataset context and any structured results into the local model. The final answer is then shown in Streamlit together with charts, forecast tables, or recommendation outputs.
