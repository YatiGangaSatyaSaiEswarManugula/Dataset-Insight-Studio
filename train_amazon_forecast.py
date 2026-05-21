import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


FILTER_COLUMNS = ["Category", "Country", "OrderStatus", "PaymentMethod", "Brand"]
GROUP_COLUMNS = ["Category", "Country", "OrderStatus", "PaymentMethod", "Brand", "ProductName", "City", "State"]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_order_date = df["OrderDate"].copy()
    normalized_raw = raw_order_date.astype(str).str.strip()
    leap_fix_mask = normalized_raw.isin(["2/29/2025", "02/29/2025", "2/29/25", "02/29/25"])
    if leap_fix_mask.any():
        raw_order_date.loc[leap_fix_mask] = raw_order_date.loc[leap_fix_mask].astype(str).str.replace("2/29/2025", "2/28/2025", regex=False)
        raw_order_date.loc[leap_fix_mask] = raw_order_date.loc[leap_fix_mask].astype(str).str.replace("02/29/2025", "02/28/2025", regex=False)
        raw_order_date.loc[leap_fix_mask] = raw_order_date.loc[leap_fix_mask].astype(str).str.replace("2/29/25", "2/28/25", regex=False)
        raw_order_date.loc[leap_fix_mask] = raw_order_date.loc[leap_fix_mask].astype(str).str.replace("02/29/25", "02/28/25", regex=False)

    df["OrderDate"] = pd.to_datetime(raw_order_date, errors="coerce")
    if df["OrderDate"].isna().any():
        retry = pd.to_datetime(raw_order_date, errors="coerce", dayfirst=True)
        if retry.notna().sum() > df["OrderDate"].notna().sum():
            df["OrderDate"] = retry

    invalid_mask = df["OrderDate"].isna()
    invalid_count = int(invalid_mask.sum())
    df.attrs["repaired_invalid_leap_dates"] = int(leap_fix_mask.sum())
    if invalid_count:
        df = df.loc[~invalid_mask].copy()
        df.attrs["dropped_invalid_order_dates"] = invalid_count
        df.attrs["invalid_order_date_examples"] = raw_order_date[invalid_mask].astype(str).head(5).tolist()
    else:
        df.attrs["dropped_invalid_order_dates"] = 0
        df.attrs["invalid_order_date_examples"] = []

    df = df.sort_values("OrderDate").reset_index(drop=True)
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    return df


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_data(df)


def build_value_maps(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    value_maps = {}
    for column in FILTER_COLUMNS:
        mapping = {}
        for value in sorted(df[column].dropna().astype(str).unique()):
            mapping[value.lower()] = value
        value_maps[column] = mapping
    return value_maps


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def extract_forecast_months(query: str, default_months: int) -> int:
    match = re.search(r"(\d+)\s*(month|months)", query.lower())
    if match:
        return max(int(match.group(1)), 1)
    if "next month" in query.lower():
        return 1
    return default_months


def extract_growth_target_percent(query: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", query.lower())
    if match:
        return float(match.group(1))
    return None


def parse_query(query: str, value_maps: dict[str, dict[str, str]], default_months: int) -> dict:
    lowered = query.lower()
    filters = {}
    normalized_query = normalize_text(query)

    recommendation_keywords = [
        "maximize profit",
        "maximise profit",
        "maximize revenue",
        "best action",
        "best actions",
        "which 3 actions",
        "recommend",
        "recommendation",
        "improve profit",
        "increase profit",
        "where should i expand",
        "where to expand",
        "expand business",
        "expand geographically",
        "business geographically",
        "best market",
        "best country",
        "which country should",
        "where should we focus",
    ]
    goal_keywords = [
        "increase sales by",
        "increase revenue by",
        "grow sales by",
        "grow revenue by",
        "reach",
        "target",
        "achieve",
        "how should i",
        "what should i do",
    ]
    forecast_keywords = [
        "predict",
        "forecast",
        "future",
        "next month",
        "next months",
        "upcoming",
        "estimate",
        "projection",
    ]
    analysis_keywords = [
        "top",
        "highest",
        "lowest",
        "most",
        "least",
        "compare",
        "comparison",
        "distribution",
        "breakdown",
        "show",
        "list",
        "analyze",
        "analyse",
        "trend",
        "trends",
    ]

    query_type = "unsupported"
    if any(keyword in lowered for keyword in recommendation_keywords):
        query_type = "recommendation"
    elif any(keyword in lowered for keyword in goal_keywords) and extract_growth_target_percent(query) is not None:
        query_type = "goal"
    elif any(keyword in lowered for keyword in forecast_keywords) or re.search(r"(\d+)\s*(month|months)", lowered):
        query_type = "forecast"
    elif any(keyword in lowered for keyword in analysis_keywords):
        query_type = "analysis"

    for column, mapping in value_maps.items():
        for value_lower, original_value in mapping.items():
            normalized_value = normalize_text(value_lower)
            if value_lower in lowered or normalized_value in normalized_query:
                filters[column] = original_value
                break

    metric = "TotalAmount"
    metric_label = "Revenue"
    if "quantity" in lowered or "units sold" in lowered or "unit sold" in lowered or "volume" in lowered:
        metric = "Quantity"
        metric_label = "Quantity"
    elif "orders" in lowered or "order count" in lowered:
        metric = "OrderCount"
        metric_label = "Orders"
    elif "profit" in lowered:
        metric = "TotalAmount"
        metric_label = "Revenue Proxy"

    months = extract_forecast_months(query, default_months)
    group_by = None
    for column in GROUP_COLUMNS:
        if normalize_text(column) in normalized_query or column.lower() in lowered:
            group_by = column
            break
    if group_by is None:
        alias_map = {
            "category": "Category",
            "categories": "Category",
            "country": "Country",
            "countries": "Country",
            "status": "OrderStatus",
            "payment": "PaymentMethod",
            "brand": "Brand",
            "product": "ProductName",
            "products": "ProductName",
            "city": "City",
            "state": "State",
        }
        for alias, column in alias_map.items():
            if alias in normalized_query:
                group_by = column
                break

    rank = "top"
    if any(word in lowered for word in ["lowest", "least", "bottom"]):
        rank = "bottom"

    if query_type == "unsupported" and (filters or metric != "TotalAmount"):
        query_type = "analysis"

    return {
        "filters": filters,
        "metric": metric,
        "metric_label": metric_label,
        "forecast_months": months,
        "growth_target_percent": extract_growth_target_percent(query),
        "raw_query": query,
        "query_type": query_type,
        "group_by": group_by,
        "rank": rank,
    }


def filter_dataframe(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    filtered = df.copy()
    for column, value in filters.items():
        filtered = filtered[filtered[column] == value]
    return filtered


def build_monthly_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "OrderCount":
        monthly_series = df.set_index("OrderDate").resample("MS").size().rename("MonthlyTarget")
    else:
        monthly_series = (
            df.set_index("OrderDate")
            .resample("MS")[metric]
            .sum()
            .rename("MonthlyTarget")
            .fillna(0)
        )
    if len(monthly_series) < 24:
        raise ValueError("Need at least 24 months of data for forecasting after applying query filters.")
    return monthly_series


def build_features(series: pd.Series) -> pd.DataFrame:
    feature_df = pd.DataFrame({"target": series})
    feature_df["lag_1"] = feature_df["target"].shift(1)
    feature_df["lag_2"] = feature_df["target"].shift(2)
    feature_df["lag_3"] = feature_df["target"].shift(3)
    feature_df["lag_6"] = feature_df["target"].shift(6)
    feature_df["lag_12"] = feature_df["target"].shift(12)
    feature_df["rolling_mean_3"] = feature_df["target"].shift(1).rolling(window=3).mean()
    feature_df["rolling_mean_6"] = feature_df["target"].shift(1).rolling(window=6).mean()
    feature_df["month"] = feature_df.index.month
    feature_df["quarter"] = feature_df.index.quarter
    feature_df["year"] = feature_df.index.year
    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)
    return feature_df.dropna()


def split_train_test(features: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(features) <= test_months:
        raise ValueError("Not enough feature rows after lagging to create a test split.")
    return features.iloc[:-test_months].copy(), features.iloc[-test_months:].copy()


def train_model(train_df: pd.DataFrame) -> tuple[RandomForestRegressor, list[str]]:
    feature_cols = [col for col in train_df.columns if col != "target"]
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["target"])
    return model, feature_cols


def recursive_forecast(model: RandomForestRegressor, history: pd.Series, steps: int) -> pd.Series:
    history = history.copy()
    predictions = []
    for _ in range(steps):
        next_date = history.index[-1] + pd.offsets.MonthBegin(1)
        feature_row = {
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_6": history.iloc[-6],
            "lag_12": history.iloc[-12],
            "rolling_mean_3": history.iloc[-3:].mean(),
            "rolling_mean_6": history.iloc[-6:].mean(),
            "month": next_date.month,
            "quarter": next_date.quarter,
            "year": next_date.year,
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12),
        }
        prediction = model.predict(pd.DataFrame([feature_row]))[0]
        predictions.append((next_date, prediction))
        history.loc[next_date] = prediction
    return pd.Series(
        data=[pred for _, pred in predictions],
        index=[date for date, _ in predictions],
        name="ForecastValue",
    )


def create_analysis_tables(df: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    delivered_df = df[df["OrderStatus"] == "Delivered"].copy()
    metric_series = pd.Series(np.ones(len(df)), index=df.index) if metric == "OrderCount" else df[metric]

    def aggregate_metric(frame: pd.DataFrame, group_cols: list[str] | str) -> pd.DataFrame:
        grouped = frame.groupby(group_cols).agg(Orders=("OrderID", "count"), QuantitySold=("Quantity", "sum"))
        grouped["MetricValue"] = grouped["Orders"] if metric == "OrderCount" else frame.groupby(group_cols)[metric].sum()
        return grouped.reset_index()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Start Date",
                "End Date",
                "Metric Used",
                "Total Metric Value",
                "Average Order Value",
                "Total Quantity Sold",
                "Delivered Order Share",
                "Unique Customers",
            ],
            "Value": [
                len(df),
                len(df.columns),
                df["OrderDate"].min().date().isoformat(),
                df["OrderDate"].max().date().isoformat(),
                metric,
                round(metric_series.sum(), 2),
                round(df["TotalAmount"].mean(), 2),
                int(df["Quantity"].sum()),
                round(df["OrderStatus"].eq("Delivered").mean() * 100, 2),
                df["CustomerID"].nunique(),
            ],
        }
    )
    monthly_metric = (
        (
            df.groupby("YearMonth").size().reset_index(name="MetricValue")
            if metric == "OrderCount"
            else df.groupby("YearMonth")[metric].sum().reset_index(name="MetricValue")
        ).sort_values("YearMonth")
    )
    category_metric = aggregate_metric(df, "Category").sort_values("MetricValue", ascending=False).reset_index(drop=True)
    country_metric = aggregate_metric(df, "Country").sort_values("MetricValue", ascending=False).reset_index(drop=True)
    status_metric = aggregate_metric(df, "OrderStatus").sort_values("Orders", ascending=False).reset_index(drop=True)
    top_products = (
        aggregate_metric(delivered_df, ["ProductID", "ProductName", "Category", "Brand"])
        .sort_values("MetricValue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    return {
        "summary": summary,
        "monthly_metric": monthly_metric,
        "category_metric": category_metric,
        "country_metric": country_metric,
        "status_metric": status_metric,
        "top_products": top_products,
    }


def generate_recommendations(df: pd.DataFrame) -> tuple[list[dict[str, str]], pd.DataFrame]:
    category_metric = (
        df.groupby("Category")
        .agg(Revenue=("TotalAmount", "sum"), Orders=("OrderID", "count"))
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    country_metric = (
        df.groupby("Country")
        .agg(Revenue=("TotalAmount", "sum"), Orders=("OrderID", "count"))
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    status_metric = (
        df.groupby("OrderStatus")
        .agg(Revenue=("TotalAmount", "sum"), Orders=("OrderID", "count"))
        .sort_values("Orders", ascending=False)
        .reset_index()
    )
    brand_metric = (
        df.groupby("Brand")
        .agg(Revenue=("TotalAmount", "sum"), AvgOrderValue=("TotalAmount", "mean"))
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )

    top_category = category_metric.iloc[0]
    top_country = country_metric.iloc[0]
    returned_cancelled = status_metric[status_metric["OrderStatus"].isin(["Returned", "Cancelled"])]
    lost_revenue = returned_cancelled["Revenue"].sum()
    lost_orders = returned_cancelled["Orders"].sum()
    top_brand = brand_metric.iloc[0]

    recommendations = [
        {
            "Action": f"Prioritize {top_category['Category']}",
            "Why": (
                f"It is the highest-revenue category with {top_category['Revenue']:,.2f} in sales "
                f"across {int(top_category['Orders']):,} orders."
            ),
            "ImpactProxy": f"Category revenue: {top_category['Revenue']:,.2f}",
        },
        {
            "Action": f"Expand campaigns in {top_country['Country']}",
            "Why": (
                f"It is the strongest market with {top_country['Revenue']:,.2f} in revenue "
                f"from {int(top_country['Orders']):,} orders."
            ),
            "ImpactProxy": f"Country revenue: {top_country['Revenue']:,.2f}",
        },
        {
            "Action": "Reduce returned and cancelled orders",
            "Why": (
                f"Returned and cancelled orders account for {lost_orders:,} orders and "
                f"{lost_revenue:,.2f} in revenue leakage."
            ),
            "ImpactProxy": f"Revenue at risk: {lost_revenue:,.2f}",
        },
    ]

    supporting_table = pd.DataFrame(
        [
            {"Area": "Top Category", "Value": top_category["Category"], "Metric": round(top_category["Revenue"], 2)},
            {"Area": "Top Country", "Value": top_country["Country"], "Metric": round(top_country["Revenue"], 2)},
            {"Area": "Top Brand", "Value": top_brand["Brand"], "Metric": round(top_brand["Revenue"], 2)},
            {"Area": "Returned+Cancelled Revenue", "Value": "Leakage", "Metric": round(lost_revenue, 2)},
        ]
    )
    return recommendations, supporting_table


def generate_goal_plan(df: pd.DataFrame, parsed_query: dict) -> dict:
    monthly_revenue = build_monthly_series(df, "TotalAmount")
    baseline_month = monthly_revenue.index[-1]
    baseline_value = float(monthly_revenue.iloc[-1])
    growth_target_percent = parsed_query["growth_target_percent"] or 0.0
    target_value = baseline_value * (1 + growth_target_percent / 100.0)
    required_lift = target_value - baseline_value

    category_revenue = (
        df.groupby("Category")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="Revenue")
    )
    country_revenue = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="Revenue")
    )
    leakage = (
        df[df["OrderStatus"].isin(["Returned", "Cancelled"])]
        .groupby("OrderStatus")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="RevenueLeakage")
    )

    top_category = category_revenue.iloc[0]
    top_country = country_revenue.iloc[0]
    leakage_value = float(leakage["RevenueLeakage"].sum()) if not leakage.empty else 0.0

    actions = [
        {
            "Action": f"Push {top_category['Category']} campaigns first",
            "Reason": f"This is the largest revenue category at {top_category['Revenue']:,.2f}.",
            "EstimatedContribution": round(required_lift * 0.45, 2),
        },
        {
            "Action": f"Concentrate growth offers in {top_country['Country']}",
            "Reason": f"This is the strongest market at {top_country['Revenue']:,.2f} in revenue.",
            "EstimatedContribution": round(required_lift * 0.35, 2),
        },
        {
            "Action": "Reduce cancellations and returns",
            "Reason": f"Revenue leakage from returned/cancelled orders is {leakage_value:,.2f}.",
            "EstimatedContribution": round(required_lift * 0.20, 2),
        },
    ]

    summary = pd.DataFrame(
        [
            {"Metric": "Baseline Month", "Value": baseline_month.date().isoformat()},
            {"Metric": "Baseline Revenue", "Value": round(baseline_value, 2)},
            {"Metric": "Target Growth %", "Value": growth_target_percent},
            {"Metric": "Target Revenue", "Value": round(target_value, 2)},
            {"Metric": "Required Revenue Lift", "Value": round(required_lift, 2)},
        ]
    )
    action_table = pd.DataFrame(actions)
    return {
        "summary": summary,
        "actions": actions,
        "action_table": action_table,
        "baseline_series": monthly_revenue.tail(12).reset_index(name="Revenue"),
    }


def create_grouped_analysis(df: pd.DataFrame, metric: str, group_by: str | None, rank: str) -> dict:
    if group_by is None:
        group_by = "Category"

    if metric == "OrderCount":
        grouped = df.groupby(group_by).size().reset_index(name="MetricValue")
    else:
        grouped = df.groupby(group_by)[metric].sum().reset_index(name="MetricValue")

    ascending = rank == "bottom"
    grouped = grouped.sort_values("MetricValue", ascending=ascending).reset_index(drop=True)
    top_n = grouped.head(10)
    return {
        "group_by": group_by,
        "table": grouped,
        "top_n": top_n,
        "headline": f"{'Lowest' if ascending else 'Top'} {group_by} by {metric}",
    }


def save_query_outputs(
    output_dir: Path,
    parsed_query: dict,
    comparison: pd.DataFrame,
    forecast_table: pd.DataFrame,
    metrics_table: pd.DataFrame,
    analysis_tables: dict[str, pd.DataFrame],
) -> Path:
    for name, table in analysis_tables.items():
        table.to_csv(output_dir / f"query_{name}.csv", index=False)
    comparison.to_csv(output_dir / "query_forecast_test_comparison.csv", index=False)
    forecast_table.to_csv(output_dir / "query_future_forecast.csv", index=False)
    metrics_table.to_csv(output_dir / "query_forecast_metrics.csv", index=False)

    lines = [
        "QUERY-BASED AMAZON ANALYSIS AND FORECAST",
        "",
        f"User Query: {parsed_query['raw_query']}",
        f"Metric: {parsed_query['metric']}",
        f"Forecast Months: {parsed_query['forecast_months']}",
        f"Applied Filters: {parsed_query['filters'] if parsed_query['filters'] else 'None'}",
        "",
        "Summary",
        analysis_tables["summary"].to_string(index=False),
        "",
        "Forecast Metrics",
        metrics_table.to_string(index=False),
        "",
        "Future Forecast",
        forecast_table.to_string(index=False),
    ]
    report_path = output_dir / "query_analysis_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def plot_forecast(
    history: pd.Series,
    test_actual: pd.Series,
    test_pred: pd.Series,
    future_forecast: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(history.index, history.values, label="Historical", linewidth=2)
    plt.plot(test_actual.index, test_actual.values, label="Test Actual", linewidth=2)
    plt.plot(test_pred.index, test_pred.values, label="Test Predicted", linewidth=2)
    plt.plot(future_forecast.index, future_forecast.values, label="Future Forecast", linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_query_forecast(df: pd.DataFrame, parsed_query: dict, test_months: int, output_dir: Path) -> None:
    filtered_df = filter_dataframe(df, parsed_query["filters"])
    if filtered_df.empty:
        raise ValueError("No rows matched the user query filters.")

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

    comparison = pd.DataFrame(
        {
            "Month": test_df.index,
            "ActualValue": test_df["target"].round(2).values,
            "PredictedValue": test_predictions.round(2).values,
        }
    )
    forecast_table = future_forecast.round(2).rename_axis("Month").reset_index(name="ForecastValue")
    metrics_table = pd.DataFrame(
        {
            "Metric": ["Train Months", "Test Months", "MAE", "RMSE", "MAPE"],
            "Value": [len(train_df), len(test_df), round(mae, 2), round(rmse, 2), round(mape, 2)],
        }
    )
    analysis_tables = create_analysis_tables(filtered_df, parsed_query["metric"])
    report_path = save_query_outputs(output_dir, parsed_query, comparison, forecast_table, metrics_table, analysis_tables)

    plot_title = f"Forecast for query: {parsed_query['raw_query']}"
    plot_forecast(
        monthly_series,
        test_df["target"],
        test_predictions,
        future_forecast,
        output_dir / "query_forecast.png",
        plot_title,
    )

    print("Query-based analysis and forecasting complete.")
    print(f"Matched rows: {len(filtered_df)}")
    print(f"Applied filters: {parsed_query['filters'] if parsed_query['filters'] else 'None'}")
    print(f"Metric used: {parsed_query['metric']}")
    print(f"Forecast months: {parsed_query['forecast_months']}")
    print(f"Report saved to: {report_path}")
    print(f"Forecast plot saved to: {output_dir / 'query_forecast.png'}")
    print("\nForecast metrics:")
    print(metrics_table.to_string(index=False))
    print("\nFuture forecast:")
    print(forecast_table.to_string(index=False))


def run_recommendation_query(df: pd.DataFrame, parsed_query: dict) -> None:
    filtered_df = filter_dataframe(df, parsed_query["filters"])
    if filtered_df.empty:
        raise ValueError("No rows matched the user query filters.")

    recommendations, supporting_table = generate_recommendations(filtered_df)
    print("Recommendation query complete.")
    print("Dataset note: profit is not available in this dataset, so recommendations use revenue and order leakage as proxies.")
    print(f"Applied filters: {parsed_query['filters'] if parsed_query['filters'] else 'None'}")
    print("\nTop actions:")
    for idx, item in enumerate(recommendations, start=1):
        print(f"{idx}. {item['Action']} | {item['Why']} | {item['ImpactProxy']}")
    print("\nSupporting metrics:")
    print(supporting_table.to_string(index=False))


def run_analysis_query(df: pd.DataFrame, parsed_query: dict) -> None:
    filtered_df = filter_dataframe(df, parsed_query["filters"])
    if filtered_df.empty:
        raise ValueError("No rows matched the user query filters.")

    analysis = create_grouped_analysis(filtered_df, parsed_query["metric"], parsed_query["group_by"], parsed_query["rank"])
    print("Analysis query complete.")
    print(f"Applied filters: {parsed_query['filters'] if parsed_query['filters'] else 'None'}")
    print(f"Group by: {analysis['group_by']}")
    print(f"Metric: {parsed_query['metric_label']}")
    print(analysis["top_n"].to_string(index=False))


def run_goal_query(df: pd.DataFrame, parsed_query: dict) -> None:
    filtered_df = filter_dataframe(df, parsed_query["filters"])
    if filtered_df.empty:
        raise ValueError("No rows matched the user query filters.")

    goal_plan = generate_goal_plan(filtered_df, parsed_query)
    print("Goal query complete.")
    print("Dataset note: this plan uses revenue as the sales target baseline.")
    print(goal_plan["summary"].to_string(index=False))
    print("\nRecommended actions:")
    print(goal_plan["action_table"].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Amazon order data and answer user-driven future prediction queries."
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).with_name("Amazon.csv")),
        help="Path to the transaction CSV file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language query such as: predict electronics sales in India for next 6 months",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=12,
        help="How many recent months to keep for evaluation.",
    )
    parser.add_argument(
        "--forecast-months",
        type=int,
        default=6,
        help="Default future months if the query does not mention a horizon.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir = Path(__file__).parent
    df = load_data(csv_path)
    value_maps = build_value_maps(df)
    parsed_query = parse_query(args.query, value_maps, args.forecast_months)
    if parsed_query["query_type"] == "recommendation":
        run_recommendation_query(df, parsed_query)
    elif parsed_query["query_type"] == "goal":
        run_goal_query(df, parsed_query)
    elif parsed_query["query_type"] == "analysis":
        run_analysis_query(df, parsed_query)
    else:
        if parsed_query["query_type"] == "forecast":
            run_query_forecast(df, parsed_query, args.test_months, output_dir)
        else:
            raise ValueError(
                "Unsupported query. Try a forecast like 'predict electronics sales in India for next 6 months', "
                "an analysis like 'top categories by revenue', or a recommendation like 'which 3 actions will maximize profit?'."
            )


if __name__ == "__main__":
    main()

