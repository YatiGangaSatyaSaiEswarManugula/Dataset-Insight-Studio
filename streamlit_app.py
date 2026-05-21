from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from local_llm_assistant import (
    answer_query_with_local_llm,
    is_amazon_schema,
    list_ollama_models,
    ollama_is_available,
)
from train_amazon_forecast import (
    build_features,
    build_monthly_series,
    build_value_maps,
    create_grouped_analysis,
    create_analysis_tables,
    filter_dataframe,
    generate_recommendations,
    generate_goal_plan,
    load_data,
    parse_query,
    prepare_data,
    recursive_forecast,
    split_train_test,
    train_model,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


APP_DIR = Path(__file__).parent
DEFAULT_CSV = APP_DIR / "Amazon.csv"


@st.cache_data
def load_default_dataframe() -> pd.DataFrame:
    return load_data(DEFAULT_CSV)


@st.cache_data
def load_uploaded_dataframe(file_bytes: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding=encoding)
            if is_amazon_schema(df):
                try:
                    return prepare_data(df)
                except Exception:
                    return df
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Unable to read the uploaded CSV.")


def format_filters(filters: dict[str, str]) -> str:
    if not filters:
        return "None"
    return ", ".join(f"{key}={value}" for key, value in filters.items())


def summarize_rule_result(results: dict) -> str:
    mode = results.get("mode")
    if mode == "generic_grouped_forecast":
        return (
            "RULE ENGINE MODE: GENERIC GROUPED FORECAST\n"
            f"Date column: {results['date_column']}\n"
            f"Metric column: {results['metric_column']}\n"
            f"Group columns: {', '.join(results['group_columns'])}\n"
            f"Forecast months: {results['forecast_months']}\n"
            f"Forecast table:\n{results['forecast_table'].head(12).to_string(index=False)}"
        )
    if mode == "generic_forecast":
        forecast_preview = results["forecast_table"].head(6).to_string(index=False)
        metrics = results["metrics"]
        return (
            "RULE ENGINE MODE: GENERIC FORECAST\n"
            f"Date column: {results['date_column']}\n"
            f"Metric column: {results['metric_column']}\n"
            f"Forecast months: {results['forecast_months']}\n"
            f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%\n"
            f"Forecast preview:\n{forecast_preview}"
        )
    if mode == "forecast":
        forecast_preview = results["forecast_table"].head(6).to_string(index=False)
        metrics = results["metrics"]
        return (
            "RULE ENGINE MODE: FORECAST\n"
            f"Filters applied: {format_filters(results['parsed_query']['filters'])}\n"
            f"Forecast months: {results['parsed_query']['forecast_months']}\n"
            f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%\n"
            f"Forecast preview:\n{forecast_preview}"
        )
    if mode == "analysis":
        analysis = results["analysis_result"]
        return (
            "RULE ENGINE MODE: ANALYSIS\n"
            f"Filters applied: {format_filters(results['parsed_query']['filters'])}\n"
            f"Headline: {analysis['headline']}\n"
            f"Top rows:\n{analysis['top_n'].head(10).to_string(index=False)}"
        )
    if mode == "recommendation":
        rec_lines = [f"{idx}. {item['Action']} | {item['Why']}" for idx, item in enumerate(results["recommendations"], start=1)]
        return (
            "RULE ENGINE MODE: RECOMMENDATION\n"
            f"Filters applied: {format_filters(results['parsed_query']['filters'])}\n"
            + "\n".join(rec_lines)
        )
    if mode == "goal":
        return (
            "RULE ENGINE MODE: GOAL PLAN\n"
            f"Filters applied: {format_filters(results['parsed_query']['filters'])}\n"
            f"Goal summary:\n{results['goal_plan']['summary'].to_string(index=False)}\n"
            f"Actions:\n{results['goal_plan']['action_table'].to_string(index=False)}"
        )
    return ""


def detect_date_range_text(df: pd.DataFrame) -> str:
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return f"{df[column].min()} to {df[column].max()}"
        if "date" in str(column).lower():
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() >= 0.7:
                return f"{converted.min().date()} to {converted.max().date()}"
    return "Not detected"


def find_generic_date_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return str(column)
    for column in df.columns:
        if "date" in str(column).lower():
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() >= 0.7:
                return str(column)
    return None


def find_generic_metric_column(df: pd.DataFrame, query: str) -> str | None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return None

    lowered = query.lower()
    preferred_keywords = [
        ("sales", ["sales", "sale"]),
        ("revenue", ["revenue"]),
        ("profit", ["profit"]),
        ("amount", ["amount", "total"]),
        ("quantity", ["quantity", "units"]),
    ]

    for _, keywords in preferred_keywords:
        for column in numeric_cols:
            column_lower = str(column).lower()
            if any(keyword in lowered for keyword in keywords) and any(keyword in column_lower for keyword in keywords):
                return str(column)

    ranked = []
    for column in numeric_cols:
        column_lower = str(column).lower()
        score = 0
        if any(term in column_lower for term in ["sales", "revenue", "profit", "amount", "total", "quantity"]):
            score += 5
        if "id" in column_lower or "row" in column_lower:
            score -= 5
        ranked.append((score, str(column)))
    ranked.sort(reverse=True)
    return ranked[0][1]


def extract_generic_forecast_months(query: str, default_months: int) -> int:
    lowered = query.lower()
    if "next quarter" in lowered or "quarter" in lowered:
        return 3
    if "next year" in lowered or "year" in lowered:
        return 12
    for token in lowered.replace("?", " ").split():
        if token.isdigit():
            return max(int(token), 1)
    if "next month" in lowered:
        return 1
    return default_months


def detect_group_columns_from_query(df: pd.DataFrame, query: str) -> list[str]:
    lowered = query.lower()
    matched: list[str] = []
    aliases = {
        "category": "Category",
        "region": "Region",
        "segment": "Segment",
        "state": "State",
        "city": "City",
        "country": "Country",
        "sub-category": "Sub-Category",
        "sub category": "Sub-Category",
        "ship mode": "Ship Mode",
    }

    for alias, canonical in aliases.items():
        if alias in lowered and canonical in df.columns and canonical not in matched:
            matched.append(canonical)

    for column in df.columns:
        col_lower = str(column).lower()
        if col_lower in lowered and column not in matched:
            matched.append(str(column))

    return matched[:2]


def detect_primary_metric_text(df: pd.DataFrame) -> tuple[str, str]:
    if "TotalAmount" in df.columns:
        return "Revenue", f"{df['TotalAmount'].sum():,.2f}"
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        return f"Sum({numeric_cols[0]})", f"{df[numeric_cols[0]].sum():,.2f}"
    return "Numeric Metric", "Not detected"


def detect_entity_metric_text(df: pd.DataFrame) -> tuple[str, str]:
    if "CustomerID" in df.columns:
        return "Customers", f"{df['CustomerID'].nunique():,}"
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        return f"Unique {object_cols[0]}", f"{df[object_cols[0]].nunique():,}"
    return "Unique Values", "Not detected"


def run_generic_forecast_from_query(df: pd.DataFrame, query: str, test_months: int, default_forecast_months: int) -> dict:
    working_df = df.copy()
    date_column = find_generic_date_column(working_df)
    metric_column = find_generic_metric_column(working_df, query)
    if date_column is None or metric_column is None:
        raise ValueError("This dataset does not have a clear date column and numeric metric column for forecasting.")

    if not pd.api.types.is_datetime64_any_dtype(working_df[date_column]):
        working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")
    working_df = working_df.dropna(subset=[date_column])
    if working_df.empty:
        raise ValueError("The detected date column could not be parsed for forecasting.")

    monthly_series = (
        working_df.set_index(date_column)
        .sort_index()
        .resample("MS")[metric_column]
        .sum()
        .rename("MonthlyTarget")
        .fillna(0)
    )
    if len(monthly_series) < 18:
        raise ValueError("Need at least 18 months of history in the uploaded dataset for generic forecasting.")

    forecast_months = extract_generic_forecast_months(query, default_forecast_months)
    features = build_features(monthly_series)
    if len(features) <= test_months:
        test_months = max(6, min(12, len(features) // 3))
    train_df, test_df = split_train_test(features, test_months)
    model, feature_cols = train_model(train_df)

    test_predictions = pd.Series(
        model.predict(test_df[feature_cols]),
        index=test_df.index,
        name="PredictedValue",
    )
    future_forecast = recursive_forecast(model, monthly_series, forecast_months)
    metrics = {
        "train_months": len(train_df),
        "test_months": len(test_df),
        "mae": mean_absolute_error(test_df["target"], test_predictions),
        "rmse": root_mean_squared_error(test_df["target"], test_predictions),
        "mape": mean_absolute_percentage_error(test_df["target"], test_predictions) * 100,
    }

    return {
        "mode": "generic_forecast",
        "date_column": date_column,
        "metric_column": metric_column,
        "history": monthly_series,
        "test_actual": test_df["target"],
        "test_pred": test_predictions,
        "future_forecast": future_forecast,
        "forecast_table": future_forecast.round(2).rename_axis("Month").reset_index(name="ForecastValue"),
        "metrics": metrics,
        "forecast_months": forecast_months,
    }


def run_generic_grouped_forecast(df: pd.DataFrame, query: str, default_forecast_months: int) -> dict:
    working_df = df.copy()
    date_column = find_generic_date_column(working_df)
    metric_column = find_generic_metric_column(working_df, query)
    group_columns = detect_group_columns_from_query(working_df, query)

    if date_column is None or metric_column is None:
        raise ValueError("This dataset does not have a clear date column and numeric metric column for grouped forecasting.")
    if not group_columns:
        raise ValueError("No grouping columns from the query matched the uploaded dataset.")

    if not pd.api.types.is_datetime64_any_dtype(working_df[date_column]):
        working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")
    working_df = working_df.dropna(subset=[date_column] + group_columns)
    if working_df.empty:
        raise ValueError("The detected date or grouping columns could not be used for grouped forecasting.")

    forecast_months = extract_generic_forecast_months(query, default_forecast_months)
    group_key = group_columns if len(group_columns) > 1 else group_columns[0]
    grouped_forecasts = []

    for keys, group_df in working_df.groupby(group_key):
        monthly_series = (
            group_df.set_index(date_column)
            .sort_index()
            .resample("MS")[metric_column]
            .sum()
            .rename("MonthlyTarget")
            .fillna(0)
        )
        if len(monthly_series) < 18:
            continue
        features = build_features(monthly_series)
        if len(features) < 12:
            continue
        train_df = features.iloc[:-6] if len(features) > 12 else features.iloc[:-3]
        if train_df.empty:
            continue
        model, _ = train_model(train_df)
        future_forecast = recursive_forecast(model, monthly_series, forecast_months)
        label = keys if isinstance(keys, str) else " | ".join(map(str, keys if isinstance(keys, tuple) else [keys]))
        forecast_total = float(future_forecast.sum())
        grouped_forecasts.append(
            {
                "Group": label,
                "ForecastTotal": round(forecast_total, 2),
                "LatestActual": round(float(monthly_series.iloc[-1]), 2),
            }
        )

    if not grouped_forecasts:
        raise ValueError("Not enough grouped history was available to build grouped forecasts for this query.")

    forecast_table = pd.DataFrame(grouped_forecasts).sort_values("ForecastTotal", ascending=False).reset_index(drop=True)
    return {
        "mode": "generic_grouped_forecast",
        "date_column": date_column,
        "metric_column": metric_column,
        "group_columns": group_columns,
        "forecast_months": forecast_months,
        "forecast_table": forecast_table,
    }


def run_forecast_from_query(df: pd.DataFrame, query: str, test_months: int, default_forecast_months: int) -> dict:
    value_maps = build_value_maps(df)
    parsed_query = parse_query(query, value_maps, default_forecast_months)
    filtered_df = filter_dataframe(df, parsed_query["filters"])
    if parsed_query["query_type"] == "unsupported":
        return {
            "mode": "unsupported",
            "parsed_query": parsed_query,
            "message": (
                "This query is outside the current rule-based parser. "
                "Try a forecast, an analysis question, or a recommendation question."
            ),
        }

    if filtered_df.empty:
        raise ValueError("No rows matched the query. Try a different category, country, brand, status, or payment method.")

    if parsed_query["query_type"] == "recommendation":
        recommendations, supporting_table = generate_recommendations(filtered_df)
        return {
            "mode": "recommendation",
            "parsed_query": parsed_query,
            "filtered_df": filtered_df,
            "recommendations": recommendations,
            "supporting_table": supporting_table,
        }

    if parsed_query["query_type"] == "goal":
        goal_plan = generate_goal_plan(filtered_df, parsed_query)
        return {
            "mode": "goal",
            "parsed_query": parsed_query,
            "filtered_df": filtered_df,
            "goal_plan": goal_plan,
        }

    if parsed_query["query_type"] == "analysis":
        analysis = create_grouped_analysis(filtered_df, parsed_query["metric"], parsed_query["group_by"], parsed_query["rank"])
        return {
            "mode": "analysis",
            "parsed_query": parsed_query,
            "filtered_df": filtered_df,
            "analysis_result": analysis,
        }

    monthly_series = build_monthly_series(filtered_df, parsed_query["metric"])
    features = build_features(monthly_series)
    train_df, test_df = split_train_test(features, test_months)
    model, feature_cols = train_model(train_df)

    test_predictions = pd.Series(
        model.predict(test_df[feature_cols]),
        index=test_df.index,
        name="PredictedValue",
    )
    future_forecast = recursive_forecast(model, monthly_series, parsed_query["forecast_months"])

    mae = mean_absolute_error(test_df["target"], test_predictions)
    rmse = root_mean_squared_error(test_df["target"], test_predictions)
    mape = mean_absolute_percentage_error(test_df["target"], test_predictions) * 100

    analysis_tables = create_analysis_tables(filtered_df, parsed_query["metric"])
    comparison = pd.DataFrame(
        {
            "Month": test_df.index,
            "ActualValue": test_df["target"].round(2).values,
            "PredictedValue": test_predictions.round(2).values,
        }
    )
    forecast_table = future_forecast.round(2).rename_axis("Month").reset_index(name="ForecastValue")
    metrics = {
        "train_months": len(train_df),
        "test_months": len(test_df),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

    return {
        "mode": "forecast",
        "parsed_query": parsed_query,
        "filtered_df": filtered_df,
        "analysis_tables": analysis_tables,
        "comparison": comparison,
        "forecast_table": forecast_table,
        "metrics": metrics,
        "history": monthly_series,
        "test_actual": test_df["target"],
        "test_pred": test_predictions,
        "future_forecast": future_forecast,
    }


def plot_forecast(history: pd.Series, test_actual: pd.Series, test_pred: pd.Series, future_forecast: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history.index, history.values, label="Historical", linewidth=2)
    ax.plot(test_actual.index, test_actual.values, label="Test Actual", linewidth=2)
    ax.plot(test_pred.index, test_pred.values, label="Test Predicted", linewidth=2)
    ax.plot(future_forecast.index, future_forecast.values, label="Future Forecast", linestyle="--", linewidth=2)
    ax.set_title("Query-Based Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bar(table: pd.DataFrame, label_col: str, value_col: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(table[label_col].astype(str), table[value_col], color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel(label_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Dataset Insight Studio", layout="wide")

st.title("Dataset Insight Studio")
st.write("Upload a dataset, analyze it, and answer user questions with rules or a local LLM.")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload an Amazon CSV", type=["csv"])
    test_months = st.slider("Test months", min_value=6, max_value=18, value=12)
    default_forecast_months = st.slider("Default forecast months", min_value=1, max_value=12, value=6)
    st.caption("If the query includes a month count like 'next 4 months', that value overrides the default.")
    st.header("Assistant Mode")
    use_local_llm = st.checkbox("Use local LLM via Ollama", value=False)
    ollama_ready = ollama_is_available()
    available_models = list_ollama_models() if ollama_ready else []
    if use_local_llm:
        if ollama_ready:
            default_model = available_models[0] if available_models else "llama3.2:3b"
            ollama_model = st.selectbox(
                "Ollama model",
                options=available_models if available_models else [default_model],
                index=0,
            )
            st.caption("Select one of the models currently detected by Ollama.")
        else:
            ollama_model = "llama3.2:3b"
            st.warning("Ollama is not running on this machine right now. The app will show setup instructions if you submit a query.")
    else:
        ollama_model = "llama3.2:3b"

try:
    if uploaded_file is not None:
        df = load_uploaded_dataframe(uploaded_file.getvalue())
        source_label = uploaded_file.name
    else:
        df = load_default_dataframe()
        source_label = DEFAULT_CSV.name
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

dataset_is_amazon = is_amazon_schema(df)
primary_metric_label, primary_metric_value = detect_primary_metric_text(df)
entity_label, entity_value = detect_entity_metric_text(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Date Range", detect_date_range_text(df))
col3.metric(primary_metric_label, primary_metric_value)
col4.metric(entity_label, entity_value)

st.caption(f"Using dataset: `{source_label}`")
if int(df.attrs.get("dropped_invalid_order_dates", 0)) > 0:
    dropped = int(df.attrs.get("dropped_invalid_order_dates", 0))
    examples = df.attrs.get("invalid_order_date_examples", [])
    example_text = ", ".join(map(str, examples)) if examples else "No examples available"
    st.warning(f"Dropped {dropped} rows with invalid OrderDate values. Example invalid dates: {example_text}")
if dataset_is_amazon:
    st.caption("Detected schema: Amazon order dataset. Both rule-based mode and local LLM mode are available.")
else:
    st.caption("Detected schema: generic CSV dataset. Local LLM mode works on this dataset; the rule-based engine is limited to the Amazon schema.")

examples = [
    "predict electronics sales in India for next 6 months",
    "predict units sold for books in Canada for next 4 months",
    "forecast delivered revenue in United States for 3 months",
    "What should I do to increase sales by 20% next month?",
    "top categories by revenue",
    "lowest countries by orders",
    "Which 3 actions will maximize profit?",
]

query = st.text_input("Enter a prediction query", value=examples[0])
selected_example = st.selectbox("Example queries", options=examples, index=0)
if st.button("Use Selected Example"):
    query = selected_example

run_clicked = st.button("Run Query", type="primary")

if run_clicked:
    if use_local_llm:
        supplemental_context = ""
        try:
            if dataset_is_amazon:
                rule_result = run_forecast_from_query(df, query, test_months, default_forecast_months)
                if rule_result.get("mode") != "unsupported":
                    supplemental_context = summarize_rule_result(rule_result)
            else:
                if " by " in query.lower():
                    generic_result = run_generic_grouped_forecast(df, query, default_forecast_months)
                else:
                    generic_result = run_generic_forecast_from_query(df, query, test_months, default_forecast_months)
                supplemental_context = summarize_rule_result(generic_result)
                results = generic_result
        except Exception:
            supplemental_context = ""

        if not ollama_ready:
            if supplemental_context:
                st.warning("Ollama is not available, so the app is falling back to the rule-based engine for this query.")
                results = rule_result
            else:
                st.subheader("Local LLM Not Available")
                st.error("Ollama is not installed or not running.")
                st.write("To enable ChatGPT-like local answers, install Ollama, pull a model, and rerun the app.")
                st.code("ollama pull llama3.2:3b")
                st.code("ollama serve")
                st.stop()
        try:
            if ollama_ready:
                llm_result = answer_query_with_local_llm(df, query, ollama_model, supplemental_context)
                st.subheader("Local LLM Answer")
                st.caption(f"Model: `{llm_result['model']}`")
                st.write(llm_result["answer"])
                with st.expander("Dataset Context Sent To Model"):
                    st.text(llm_result["context"])
                if llm_result["supplemental_context"]:
                    with st.expander("Structured Rule Context Sent To Model"):
                        st.text(llm_result["supplemental_context"])
                st.stop()
        except Exception as exc:
            if supplemental_context:
                st.warning(f"Local LLM failed, so the app is falling back to the rule-based engine. Details: {exc}")
                results = rule_result
            else:
                st.error(str(exc))
                st.stop()

    if not use_local_llm or ("results" not in locals()):
        if not dataset_is_amazon:
            try:
                if " by " in query.lower():
                    results = run_generic_grouped_forecast(df, query, default_forecast_months)
                else:
                    results = run_generic_forecast_from_query(df, query, test_months, default_forecast_months)
            except Exception:
                st.subheader("Rule Engine Unavailable For This Dataset")
                st.error("The current rule-based engine is limited for this uploaded dataset. Enable 'Use local LLM via Ollama' for broader answers.")
                st.stop()
        try:
            if dataset_is_amazon:
                results = run_forecast_from_query(df, query, test_months, default_forecast_months)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    parsed_query = results["parsed_query"]
    if results["mode"] == "unsupported":
        st.subheader("Unsupported Query")
        st.error(results["message"])
        st.write("Try one of these patterns:")
        st.code("predict electronics sales in India for next 6 months")
        st.code("top categories by revenue")
        st.code("which 3 actions will maximize profit?")
        st.stop()

    if results["mode"] == "recommendation":
        st.subheader("Query Understanding")
        q1, q2, q3 = st.columns(3)
        q1.metric("Mode", "Recommendation")
        q2.metric("Matched Rows", f"{len(results['filtered_df']):,}")
        q3.metric("Filters", format_filters(parsed_query["filters"]))

        st.warning("This dataset does not contain a true profit column, so the app is using revenue and order leakage as profit proxies.")

        st.subheader("Top 3 Recommended Actions")
        for idx, item in enumerate(results["recommendations"], start=1):
            st.markdown(f"**{idx}. {item['Action']}**")
            st.write(item["Why"])
            st.caption(item["ImpactProxy"])

        st.subheader("Supporting Metrics")
        st.dataframe(results["supporting_table"], use_container_width=True)
        st.download_button(
            "Download Supporting Metrics CSV",
            data=results["supporting_table"].to_csv(index=False).encode("utf-8"),
            file_name="recommendation_supporting_metrics.csv",
            mime="text/csv",
        )
        st.stop()

    if results["mode"] == "goal":
        goal_plan = results["goal_plan"]
        st.subheader("Goal Planning")
        q1, q2, q3 = st.columns(3)
        q1.metric("Mode", "Goal Plan")
        q2.metric("Matched Rows", f"{len(results['filtered_df']):,}")
        q3.metric("Filters", format_filters(parsed_query["filters"]))

        st.info("This goal plan uses the latest observed monthly revenue as the baseline and converts your target into a required revenue lift.")
        st.dataframe(goal_plan["summary"], use_container_width=True)

        st.subheader("Recommended Actions")
        for idx, item in enumerate(goal_plan["actions"], start=1):
            st.markdown(f"**{idx}. {item['Action']}**")
            st.write(item["Reason"])
            st.caption(f"Suggested contribution toward target: {item['EstimatedContribution']:,.2f}")

        st.subheader("Recent Revenue Baseline")
        baseline_df = goal_plan["baseline_series"].copy()
        st.dataframe(baseline_df, use_container_width=True)
        st.pyplot(plot_bar(baseline_df, "OrderDate", "Revenue", "Last 12 Months Revenue"))
        st.download_button(
            "Download Goal Plan CSV",
            data=goal_plan["action_table"].to_csv(index=False).encode("utf-8"),
            file_name="goal_plan_actions.csv",
            mime="text/csv",
        )
        st.stop()

    if results["mode"] == "analysis":
        analysis = results["analysis_result"]
        st.subheader("Query Understanding")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Mode", "Analysis")
        q2.metric("Metric", parsed_query["metric_label"])
        q3.metric("Matched Rows", f"{len(results['filtered_df']):,}")
        q4.metric("Filters", format_filters(parsed_query["filters"]))

        st.subheader(analysis["headline"])
        st.dataframe(analysis["top_n"], use_container_width=True)
        st.pyplot(plot_bar(analysis["top_n"], analysis["group_by"], "MetricValue", analysis["headline"]))
        st.download_button(
            "Download Analysis CSV",
            data=analysis["table"].to_csv(index=False).encode("utf-8"),
            file_name="analysis_results.csv",
            mime="text/csv",
        )
        st.stop()

    if results["mode"] == "generic_grouped_forecast":
        st.subheader("Grouped Forecast")
        q1, q2, q3 = st.columns(3)
        q1.metric("Mode", "Grouped Forecast")
        q2.metric("Metric Column", results["metric_column"])
        q3.metric("Group Columns", ", ".join(results["group_columns"]))
        st.dataframe(results["forecast_table"], use_container_width=True)
        st.download_button(
            "Download Grouped Forecast CSV",
            data=results["forecast_table"].to_csv(index=False).encode("utf-8"),
            file_name="grouped_forecast.csv",
            mime="text/csv",
        )
        st.stop()

    if results["mode"] == "generic_forecast":
        metrics = results["metrics"]
        st.subheader("Generic Forecast")
        q1, q2, q3 = st.columns(3)
        q1.metric("Date Column", results["date_column"])
        q2.metric("Metric Column", results["metric_column"])
        q3.metric("Forecast Months", results["forecast_months"])
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['mae']:,.2f}")
        m2.metric("RMSE", f"{metrics['rmse']:,.2f}")
        m3.metric("MAPE", f"{metrics['mape']:.2f}%")
        st.pyplot(plot_forecast(results["history"], results["test_actual"], results["test_pred"], results["future_forecast"]))
        st.dataframe(results["forecast_table"], use_container_width=True)
        st.stop()

    analysis_tables = results["analysis_tables"]
    metrics = results["metrics"]

    st.subheader("Query Understanding")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Metric", parsed_query["metric"])
    q2.metric("Forecast Months", parsed_query["forecast_months"])
    q3.metric("Matched Rows", f"{len(results['filtered_df']):,}")
    q4.metric("Filters", format_filters(parsed_query["filters"]))

    st.subheader("Forecast Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Train Months", f"{metrics['train_months']}")
    m2.metric("Test Months", f"{metrics['test_months']}")
    m3.metric("MAE", f"{metrics['mae']:,.2f}")
    m4.metric("RMSE", f"{metrics['rmse']:,.2f}")
    m5.metric("MAPE", f"{metrics['mape']:.2f}%")

    st.subheader("Forecast Chart")
    st.pyplot(
        plot_forecast(
            results["history"],
            results["test_actual"],
            results["test_pred"],
            results["future_forecast"],
        )
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Future Forecast")
        st.dataframe(results["forecast_table"], use_container_width=True)
        st.download_button(
            "Download Forecast CSV",
            data=results["forecast_table"].to_csv(index=False).encode("utf-8"),
            file_name="future_forecast.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Test vs Predicted")
        st.dataframe(results["comparison"], use_container_width=True)
        st.download_button(
            "Download Comparison CSV",
            data=results["comparison"].to_csv(index=False).encode("utf-8"),
            file_name="forecast_test_comparison.csv",
            mime="text/csv",
        )

    st.subheader("Analysis Summary")
    st.dataframe(analysis_tables["summary"], use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(plot_bar(analysis_tables["category_metric"], "Category", "MetricValue", "Category Breakdown"))
    with chart_right:
        st.pyplot(plot_bar(analysis_tables["country_metric"], "Country", "MetricValue", "Country Breakdown"))

    st.subheader("Detailed Tables")
    st.dataframe(analysis_tables["top_products"], use_container_width=True)
    st.dataframe(analysis_tables["status_metric"], use_container_width=True)

else:
    st.info("Enter a query and click 'Run Analysis and Forecast' to generate results.")

