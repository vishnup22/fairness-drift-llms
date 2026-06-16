from src.visualization import generate_all_visualizations


def generate_artifacts(df_clean, metrics):
    model_version_df = metrics.get("model_version_details", None)
    return generate_all_visualizations(df_clean, model_version_df)


def print_run_summary(df_raw, df_clean, metrics, total_filtered):
    print("SUMMARY:")
    total_rows = len(df_raw)
    print(
        f"\n Total queries: {total_rows} | "
        f"filtered: {total_filtered} ({total_filtered / max(total_rows, 1):.2%}) | "
        f"clean: {len(df_clean)}"
    )
    print(f" Unique models: {df_clean['model'].nunique()}")
    print(f" Providers tested: {df_clean['provider'].nunique()}")
    print(f" Datasets tested: {df_clean['dataset'].nunique()}")

    print("\n Overall Scores:")
    print(f"  Mean Sentiment Score: {metrics['mean_sentiment']:.4f} (+/-{metrics['std_sentiment']:.4f})")
    print(f"  Mean Toxicity Score: {metrics['mean_toxicity']:.4f} (+/-{metrics['std_toxicity']:.4f})")
    print(f"  Mean Stereotype Score: {metrics['mean_stereotype']:.4f} (+/-{metrics['std_stereotype']:.4f})")

    model_version_df = metrics.get("model_version_details", None)
    if model_version_df is not None and len(model_version_df) > 0:
        _print_top_models(model_version_df)

    print("BENCHMARK COMPLETE")


def _print_top_models(model_version_df):
    print("\n Top 5 Models by Sentiment Score:")
    top_sentiment = model_version_df.nlargest(5, "sentiment_mean")[
        ["model", "sentiment_mean", "toxicity_mean", "stereotype_mean"]
    ]
    for _, row in top_sentiment.iterrows():
        print(
            f"  {row['model']}: Sentiment={row['sentiment_mean']:.4f}, "
            f"Toxicity={row['toxicity_mean']:.4f}, "
            f"Stereotype={row['stereotype_mean']:.4f}"
        )

    print("\n Top 5 Models by Lowest Toxicity:")
    top_toxicity = model_version_df.nsmallest(5, "toxicity_mean")[
        ["model", "sentiment_mean", "toxicity_mean", "stereotype_mean"]
    ]
    for _, row in top_toxicity.iterrows():
        print(
            f"  {row['model']}: Sentiment={row['sentiment_mean']:.4f}, "
            f"Toxicity={row['toxicity_mean']:.4f}, "
            f"Stereotype={row['stereotype_mean']:.4f}"
        )

    print("\n Top 5 Models by Lowest Stereotype Score:")
    top_stereotype = model_version_df.nsmallest(5, "stereotype_mean")[
        ["model", "sentiment_mean", "toxicity_mean", "stereotype_mean"]
    ]
    for _, row in top_stereotype.iterrows():
        print(
            f"  {row['model']}: Sentiment={row['sentiment_mean']:.4f}, "
            f"Toxicity={row['toxicity_mean']:.4f}, "
            f"Stereotype={row['stereotype_mean']:.4f}"
        )
