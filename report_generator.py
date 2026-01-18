import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict
import json
import numpy as np
import os
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Font Setup (Korean Support)
import matplotlib.font_manager as fm
import platform

system_name = platform.system()
if system_name == "Darwin":  # Mac
    plt.rcParams["font.family"] = "AppleGothic"
elif system_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    # Linux/Server - try to find a font or use default
    pass
plt.rcParams["axes.unicode_minus"] = False


def generate_evaluation_report(
    test_results: List[Dict], output_path: str = "./reports", format: str = "html"
) -> str:
    """
    Generates a visualized evaluation report from test results.
    """
    if not test_results:
        logging.warning("No test results provided. Skipping report generation.")
        return ""

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "test_id": res.get("test_id", idx),
                "model": res.get("model_name", "Unknown"),
                "case_type": res.get("case_type", "Normal"),
                "execution_time": res.get("execution_time", 0),
                "input_tokens": res.get("input_tokens", 0),
                "output_tokens": res.get("output_tokens", 0),
                "total_tokens": res.get("input_tokens", 0)
                + res.get("output_tokens", 0),
                "input": res.get("input", {}),
                # Handle output text whether it's dict or string
                "output_text": json.dumps(
                    res.get("output", ""), ensure_ascii=False, indent=2
                )
                if isinstance(res.get("output"), dict)
                else str(res.get("output", "")),
                **res.get("metrics", {}),
            }
            for idx, res in enumerate(test_results)
        ]
    )

    # 1. Image Generation
    img_paths = {}

    # a) Overall Score Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="model",
            y="overall_score",
            hue="model",
            palette="viridis",
            errorbar=None,
        )
        plt.axhline(0.7, color="r", linestyle="--", label="Pass Criteria (0.7)")
        plt.title("Model Overall Score Comparison")
        plt.ylim(0, 1.1)
        bar_path = os.path.join(output_path, f"images/comparison_bar_{timestamp}.png")
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()
        img_paths["bar"] = os.path.relpath(bar_path, output_path)
    except Exception as e:
        logging.error(f"Failed to generate bar chart: {e}")

    # b) Radar Chart (Metrics Profile)
    metrics_cols = [
        "json_schema_compliance",
        "field_coverage",
        "classification_accuracy",
        "reasoning_depth",
        "discussion_readiness",
        "specificity",
        "consistency",
        "extra_text_parsing",
    ]
    metrics_cols = [c for c in metrics_cols if c in df.columns]

    if not df.empty:
        try:
            avg_metrics = df.groupby("model")[metrics_cols].mean()

            plt.figure(figsize=(10, 10))
            categories = metrics_cols
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += [angles[0]]

            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)

            for model_name in avg_metrics.index:
                values = avg_metrics.loc[model_name].values.flatten().tolist()
                if len(values) == N:
                    values += [values[0]]
                    ax.plot(
                        angles, values, linewidth=1, linestyle="solid", label=model_name
                    )
                    ax.fill(angles, values, alpha=0.1)

            plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
            plt.title("Model Metrics Profile (Radar Chart)")
            radar_path = os.path.join(
                output_path, f"images/radar_chart_{timestamp}.png"
            )
            plt.savefig(radar_path, bbox_inches="tight")
            plt.close()
            img_paths["radar"] = os.path.relpath(radar_path, output_path)
        except Exception as e:
            logging.error(f"Failed to generate radar chart: {e}")

    # c) Heatmap
    try:
        avg_metrics = df.groupby("model")[metrics_cols].mean()
        if not avg_metrics.empty:
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                avg_metrics, annot=True, cmap="RdYlGn", vmin=0, vmax=1, fmt=".2f"
            )
            plt.title("Model x Metrics Heatmap")
            heatmap_path = os.path.join(output_path, f"images/heatmap_{timestamp}.png")
            plt.savefig(heatmap_path, bbox_inches="tight")
            plt.close()
            img_paths["heatmap"] = os.path.relpath(heatmap_path, output_path)
    except Exception as e:
        logging.error(f"Failed to generate heatmap: {e}")

    # 3. HTML Report
    total_cases = len(df)
    models = df["model"].unique().tolist()
    global_pass_rate = (df["overall_score"] >= 0.7).mean() * 100
    avg_time = df["execution_time"].mean()

    # Model Comparison Table
    model_stats = df.groupby("model").agg(
        {
            "overall_score": "mean",
            "execution_time": "mean",
            "input_tokens": "mean",
            "output_tokens": "mean",
            "total_tokens": "mean",
        }
    )
    # Calculate Success Rate per model
    success_rates = df.groupby("model")["overall_score"].apply(
        lambda x: (x >= 0.7).mean() * 100
    )
    model_stats["success_rate"] = success_rates
    model_stats = model_stats.reset_index()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Persona Evaluation Report</title>
        <style>
            body {{ font-family: 'AppleGothic', 'Malgun Gothic', 'Noto Sans KR', sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .summary-box {{ display: flex; justify-content: space-around; background: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .stat {{ text-align: center; }}
            .value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; table-layout: fixed; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; word-wrap: break-word; }}
            th {{ background-color: #f8f9fa; }}
            .pass {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
            .small-text {{ font-size: 0.85em; color: #666; }}
            .output-box {{ 
                max-height: 200px; 
                overflow-y: auto; 
                background: #f8f9fa; 
                padding: 10px; 
                border-radius: 4px; 
                font-family: monospace; 
                font-size: 0.8em; 
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Persona Generation Evaluation Report</h1>
            <p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <div class="stat"><div class="value">{len(models)}</div><div class="label">Models Evaluated</div></div>
                <div class="stat"><div class="value">{total_cases}</div><div class="label">Total Cases</div></div>
                <div class="stat"><div class="value">{global_pass_rate:.1f}%</div><div class="label">Global Pass Rate</div></div>
                <div class="stat"><div class="value">{avg_time:.2f}s</div><div class="label">Avg Latency</div></div>
            </div>

            <h2>1. Visual Analysis</h2>
            <div class="grid">
                <div>
                    <h3>Overall Comparison</h3>
                    <img src="{img_paths.get("bar", "")}" alt="Bar Chart">
                </div>
                <div>
                    <h3>Metrics Profile</h3>
                    <img src="{img_paths.get("radar", "")}" alt="Radar Chart">
                </div>
            </div>
            <div>
                <h3>Detailed Metrics Heatmap</h3>
                <img src="{img_paths.get("heatmap", "")}" alt="Heatmap">
            </div>

            <h2>2. Model Performance Summary</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th style="width: 15%">Model</th>
                        <th>Avg Score</th>
                        <th>Success Rate (%)</th>
                        <th>Avg Latency (s)</th>
                        <th>Avg Input Tokens</th>
                        <th>Avg Output Tokens</th>
                        <th>Avg Total Tokens</th>
                    </tr>
                </thead>
                <tbody>
    """

    for _, row in model_stats.iterrows():
        html_content += f"""
            <tr>
                <td>{row["model"]}</td>
                <td>{row["overall_score"]:.2f}</td>
                <td>{row["success_rate"]:.1f}%</td>
                <td>{row["execution_time"]:.2f}</td>
                <td>{int(row["input_tokens"])}</td>
                <td>{int(row["output_tokens"])}</td>
                <td>{int(row["total_tokens"])}</td>
            </tr>
        """

    html_content += f"""
                </tbody>
            </table>
            
            <h3>Metric Breakdown</h3>
            {avg_metrics.to_html(classes="table", float_format="%.2f")}

            <h2>3. Detailed Test Results</h2>
            <table>
                <tr>
                    <th style="width: 5%">ID</th>
                    <th style="width: 10%">Model</th>
                    <th style="width: 8%">Type</th>
                    <th style="width: 5%">Score</th>
                    <th style="width: 5%">Status</th>
                    <th style="width: 25%">Input Case</th>
                    <th style="width: 32%">Generated Output</th>
                    <th style="width: 10%">Tokens (In/Out)</th>
                </tr>
                {
        "".join(
            [
                f"<tr>"
                f"<td>{row['test_id']}</td>"
                f"<td>{row['model']}</td>"
                f"<td>{row['case_type']}</td>"
                f"<td>{row['overall_score']:.2f}</td>"
                f"<td class='{'pass' if row['overall_score'] >= 0.7 else 'fail'}'>{'PASS' if row['overall_score'] >= 0.7 else 'FAIL'}</td>"
                f"<td><div class='output-box'>"
                f"<b>Name:</b> {row['input'].get('name')}<br>"
                f"<b>Profile:</b> {row['input'].get('gender')}, {row['input'].get('age_group')}<br>"
                f"<b>Allergies:</b> {row['input'].get('allergies')}<br>"
                f"<b>Pref:</b> {row['input'].get('preferred_food_categories')}<br>"
                f"<b>Extra:</b> {row['input'].get('extra_text')}"
                f"</div></td>"
                f"<td><div class='output-box'>{row['output_text']}</div></td>"
                f"<td class='small-text'>{int(row.get('input_tokens', 0))} / {int(row.get('output_tokens', 0))}</td>"
                f"</tr>"
                for _, row in df.iterrows()
            ]
        )
    }
            </table>
        </div>
    </body>
    </html>
    """

    report_file = os.path.join(output_path, f"persona_eval_report_{timestamp}.html")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logging.info(f"Report saved to: {report_file}")
    return report_file
