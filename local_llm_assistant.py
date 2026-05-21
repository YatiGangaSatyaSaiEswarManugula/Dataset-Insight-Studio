import json
import urllib.error
import urllib.request

import pandas as pd


OLLAMA_URL = "http://127.0.0.1:11434"
AMAZON_REQUIRED_COLUMNS = {
    "OrderDate",
    "TotalAmount",
    "Category",
    "Country",
    "OrderStatus",
    "CustomerID",
    "ProductID",
}


def ollama_is_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return [model["name"] for model in payload.get("models", [])]
    except Exception:
        return []


def is_amazon_schema(df: pd.DataFrame) -> bool:
    return AMAZON_REQUIRED_COLUMNS.issubset(set(df.columns))


def _coerce_date_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    date_cols = []
    for column in df.columns:
        if df[column].dtype == "object" or "date" in column.lower():
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() >= 0.7:
                df[column] = converted
                date_cols.append(column)
    return df, date_cols


def build_generic_dataset_context(df: pd.DataFrame) -> str:
    profiled_df, date_cols = _coerce_date_columns(df)
    numeric_cols = profiled_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = profiled_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = [col for col in date_cols if col in profiled_df.columns]

    lines = [
        "GENERIC DATASET SUMMARY",
        f"Rows: {len(profiled_df)}",
        f"Columns: {len(profiled_df.columns)}",
        f"Column names: {', '.join(map(str, profiled_df.columns.tolist()))}",
        f"Numeric columns: {', '.join(numeric_cols[:12]) if numeric_cols else 'None'}",
        f"Categorical columns: {', '.join(categorical_cols[:12]) if categorical_cols else 'None'}",
        f"Date-like columns: {', '.join(date_cols[:12]) if date_cols else 'None'}",
        "",
        "MISSING VALUES",
        profiled_df.isna().sum().sort_values(ascending=False).head(10).to_string(),
        "",
    ]

    if numeric_cols:
        lines.extend(
            [
                "NUMERIC SUMMARY",
                profiled_df[numeric_cols].describe().round(2).transpose().head(10).to_string(),
                "",
            ]
        )

    if categorical_cols:
        top_cat_sections = []
        for column in categorical_cols[:3]:
            top_cat_sections.append(f"{column}\n{profiled_df[column].astype(str).value_counts().head(5).to_string()}")
        lines.extend(["TOP CATEGORICAL VALUES", "\n\n".join(top_cat_sections), ""])

    if date_cols and numeric_cols:
        primary_date = date_cols[0]
        primary_metric = numeric_cols[0]
        monthly = (
            profiled_df.dropna(subset=[primary_date])
            .set_index(primary_date)
            .resample("MS")[primary_metric]
            .sum()
            .tail(12)
            .round(2)
        )
        if not monthly.empty:
            lines.extend(
                [
                    f"LAST 12 MONTHS OF {primary_metric} BY {primary_date}",
                    monthly.to_string(),
                    "",
                ]
            )

    lines.extend(
        [
            "DATASET RULES",
            "- Answer only from the dataset context provided.",
            "- If the dataset does not support the exact request, say that clearly and provide the closest supported insight.",
            "- If the user asks for recommendations, ground them in observed patterns from the dataset.",
        ]
    )
    return "\n".join(lines)


def build_amazon_dataset_context(df: pd.DataFrame) -> str:
    monthly_revenue = (
        df.set_index("OrderDate")
        .resample("MS")["TotalAmount"]
        .sum()
        .tail(12)
        .round(2)
    )
    top_categories = (
        df.groupby("Category")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .round(2)
    )
    top_countries = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .round(2)
    )
    order_status = (
        df.groupby("OrderStatus")
        .agg(Orders=("OrderID", "count"), Revenue=("TotalAmount", "sum"))
        .sort_values("Orders", ascending=False)
        .round(2)
    )

    lines = [
        "AMAZON DATASET SUMMARY",
        f"Rows: {len(df)}",
        f"Date range: {df['OrderDate'].min().date()} to {df['OrderDate'].max().date()}",
        f"Total revenue: {df['TotalAmount'].sum():.2f}",
        f"Average order value: {df['TotalAmount'].mean():.2f}",
        f"Unique customers: {df['CustomerID'].nunique()}",
        f"Unique products: {df['ProductID'].nunique()}",
        "",
        "TOP CATEGORIES BY REVENUE",
        top_categories.to_string(),
        "",
        "TOP COUNTRIES BY REVENUE",
        top_countries.to_string(),
        "",
        "ORDER STATUS SUMMARY",
        order_status.to_string(),
        "",
        "LAST 12 MONTHS REVENUE",
        monthly_revenue.to_string(),
        "",
        "DATASET NOTES",
        "- There is no true profit column in this dataset.",
        "- Use revenue and cancelled/returned order leakage as profit proxies when needed.",
        "- Answer only from the dataset context. If the dataset does not support a claim, say so clearly.",
    ]
    return "\n".join(lines)


def build_dataset_context(df: pd.DataFrame) -> str:
    if is_amazon_schema(df):
        return build_amazon_dataset_context(df)
    return build_generic_dataset_context(df)


def build_llm_prompt(query: str, dataset_context: str, supplemental_context: str = "") -> str:
    extra_context = f"\nADDITIONAL STRUCTURED CONTEXT\n{supplemental_context}\n" if supplemental_context else ""
    return (
        "You are a data analyst assistant working only from the provided dataset context.\n"
        "Rules:\n"
        "1. Answer the user's question directly and clearly.\n"
        "2. Use only the provided dataset context.\n"
        "3. If the dataset does not support the exact request, say that clearly and give the closest supported insight.\n"
        "4. If the user asks for actions or strategy, give practical recommendations grounded in the data.\n"
        "5. Mention assumptions briefly when the schema is ambiguous.\n\n"
        f"{dataset_context}\n"
        f"{extra_context}\n"
        f"USER QUESTION:\n{query}\n\n"
        "ANSWER:"
    )


def query_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("response", "").strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama request failed: {detail or exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Ollama: {exc}") from exc


def answer_query_with_local_llm(df: pd.DataFrame, query: str, model: str, supplemental_context: str = "") -> dict:
    context = build_dataset_context(df)
    prompt = build_llm_prompt(query, context, supplemental_context)
    answer = query_ollama(model, prompt)
    return {
        "answer": answer,
        "context": context,
        "supplemental_context": supplemental_context,
        "model": model,
    }
