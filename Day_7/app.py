"""
Student Performance Dashboard
Day 7 mini project - MLB Internship

A small analytics dashboard for exploring cleaned student score data.
Charts are built with Matplotlib and Seaborn.

Layout:
  - Overview: top performers, students needing improvement, and the
    two summary charts, all in one place.
  - Charts: the deeper comparison charts (scores vs each other,
    spread, performance mix).
  - Full Dataset: the raw table, with its own program filter and a
    sort control (Average Score / Performance / Attendance).
"""

import numpy as np
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------

DATA_PATH = "cleaned_student_performance.csv"
df = pd.read_csv(DATA_PATH)

SUBJECT_COLS = ["Python_Score", "Mathematics_Score", "Statistics_Score", "ML_Score"]

df["Average_Score"] = df[SUBJECT_COLS].mean(axis=1).round(2)


def classify_performance(score):
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Average"
    else:
        return "Needs Improvement"


df["Performance"] = df["Average_Score"].apply(classify_performance)

# Program / track column - detect it, or fall back to a single group
PROGRAM_COL = None
for c in df.columns:
    if c.strip().lower() in ("program", "course", "track", "department", "branch"):
        PROGRAM_COL = c
        break

if PROGRAM_COL is None:
    df["Program"] = "General"
    PROGRAM_COL = "Program"

PROGRAM_OPTIONS = ["All"] + sorted(df[PROGRAM_COL].dropna().unique().tolist())

# Attendance column - detect it, or generate a placeholder so the
# dashboard still works. Replace this with your real column if your
# CSV already tracks attendance under a different name.
ATTENDANCE_COL = None
for c in df.columns:
    if "attendance" in c.strip().lower():
        ATTENDANCE_COL = c
        break

if ATTENDANCE_COL is None:
    rng = np.random.default_rng(42)
    df["Attendance_Percentage"] = rng.integers(70, 100, size=len(df))
    ATTENDANCE_COL = "Attendance_Percentage"

SUBJECT_LABELS = {
    "Python_Score": "Python",
    "Mathematics_Score": "Mathematics",
    "Statistics_Score": "Statistics",
    "ML_Score": "Machine Learning",
}

PERFORMANCE_ORDER = ["Excellent", "Good", "Average", "Needs Improvement"]

# Semantic colors - same meaning used consistently across every chart
PERFORMANCE_COLORS = {
    "Excellent": "#2f9e44",
    "Good": "#1971c2",
    "Average": "#f08c00",
    "Needs Improvement": "#e03131",
}

# One distinct color per subject, used consistently across charts
SUBJECT_COLORS = {
    "Python_Score": "#5c7cfa",
    "Mathematics_Score": "#12b886",
    "Statistics_Score": "#f76707",
    "ML_Score": "#ae3ec9",
}

ACCENT = "#3f5d8a"
CARD_SHADES = ["#1f2937", "#3f5d8a", "#64748b", "#94a3b8"]


# ---------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------

def build_kpi_html(data: pd.DataFrame) -> str:
    total_students = len(data)
    class_avg = round(data["Average_Score"].mean(), 1)
    avg_attendance = round(data[ATTENDANCE_COL].mean(), 1)

    top_row = data.loc[data["Average_Score"].idxmax()]
    top_performer_name = top_row["Student_Name"]
    top_performer_score = top_row["Average_Score"]

    cards = [
        ("Total Students", str(total_students), None),
        ("Class Average", str(class_avg), None),
        ("Average Attendance", f"{avg_attendance}%", None),
        ("Top Performer", top_performer_name, f"{top_performer_score} avg"),
    ]

    html = "<div class='kpi-row'>"
    for i, (label, value, sub) in enumerate(cards):
        sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
        html += f"""
        <div class='kpi-card' style='border-top-color:{CARD_SHADES[i]};'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            {sub_html}
        </div>
        """
    html += "</div>"
    return html


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------

def chart_subject_averages(data: pd.DataFrame):
    subj_avg = data[SUBJECT_COLS].mean().reset_index()
    subj_avg.columns = ["Subject", "Average"]
    bar_colors = [SUBJECT_COLORS[s] for s in subj_avg["Subject"]]
    subj_avg["Subject"] = subj_avg["Subject"].map(SUBJECT_LABELS)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(subj_avg["Subject"], subj_avg["Average"], color=bar_colors, width=0.55)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("Average Score by Subject")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 100)
    plt.close(fig)
    return fig


def chart_score_distribution(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data["Average_Score"], bins=10, color="#12b886", ax=ax)
    ax.set_title("Distribution of Average Scores")
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Number of Students")
    plt.close(fig)
    return fig


def chart_performance_breakdown(data: pd.DataFrame):
    counts = data["Performance"].value_counts().reindex(PERFORMANCE_ORDER).fillna(0)
    bar_colors = [PERFORMANCE_COLORS[label] for label in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(counts.index, counts.values, color=bar_colors, height=0.5)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.1, i, str(int(v)), va="center", fontsize=9)
    ax.set_title("Students by Performance Category")
    ax.set_xlabel("Number of Students")
    plt.close(fig)
    return fig


def chart_performance_pie(data: pd.DataFrame):
    counts = data["Performance"].value_counts().reindex(PERFORMANCE_ORDER).fillna(0)
    counts = counts[counts > 0]
    pie_colors = [PERFORMANCE_COLORS[label] for label in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white"),
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
        t.set_fontsize(9)
    ax.set_title("Performance Category Share")
    plt.close(fig)
    return fig


def chart_python_vs_ml(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in PERFORMANCE_ORDER:
        subset = data[data["Performance"] == label]
        if subset.empty:
            continue
        ax.scatter(subset["Python_Score"], subset["ML_Score"],
                   color=PERFORMANCE_COLORS[label], label=label,
                   s=70, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.set_title("Python Score vs Machine Learning Score")
    ax.set_xlabel("Python Score")
    ax.set_ylabel("Machine Learning Score")
    ax.legend(title="Performance", fontsize=8, title_fontsize=9)
    plt.close(fig)
    return fig


def chart_subject_spread(data: pd.DataFrame):
    melted = data.melt(
        id_vars=["Student_Name"], value_vars=SUBJECT_COLS,
        var_name="Subject", value_name="Score",
    )
    palette = {SUBJECT_LABELS[k]: v for k, v in SUBJECT_COLORS.items()}
    melted["Subject"] = melted["Subject"].map(SUBJECT_LABELS)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.boxplot(x="Subject", y="Score", hue="Subject", data=melted,
                palette=palette, legend=False, width=0.5, ax=ax)
    ax.set_title("Score Spread Across Subjects")
    ax.set_xlabel("")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------
# Tables
# ---------------------------------------------------------------

def get_top5(data: pd.DataFrame) -> pd.DataFrame:
    cols = ["Student_Name", PROGRAM_COL, "Average_Score", "Performance"]
    return (
        data.sort_values("Average_Score", ascending=False)
        .head(5)[cols]
        .reset_index(drop=True)
        .rename(columns={PROGRAM_COL: "Program"})
    )


def get_needs_improvement(data: pd.DataFrame) -> pd.DataFrame:
    cols = ["Student_Name", PROGRAM_COL, "Average_Score", "Performance"]
    return (
        data[data["Performance"] == "Needs Improvement"][cols]
        .sort_values("Average_Score")
        .reset_index(drop=True)
        .rename(columns={PROGRAM_COL: "Program"})
    )


SORT_OPTIONS = ["Average Score", "Performance", "Attendance"]


def get_full_dataset(program: str, sort_by: str) -> pd.DataFrame:
    data = df if program == "All" else df[df[PROGRAM_COL] == program]
    data = data.copy()

    if sort_by == "Average Score":
        data = data.sort_values("Average_Score", ascending=False)
    elif sort_by == "Performance":
        order = {label: i for i, label in enumerate(PERFORMANCE_ORDER)}
        data = data.assign(_rank=data["Performance"].map(order)).sort_values("_rank").drop(columns="_rank")
    elif sort_by == "Attendance":
        data = data.sort_values(ATTENDANCE_COL, ascending=False)

    display_cols = ["Student_Name", PROGRAM_COL] + SUBJECT_COLS + [ATTENDANCE_COL, "Average_Score", "Performance"]
    out = data[display_cols].rename(columns={
        PROGRAM_COL: "Program",
        ATTENDANCE_COL: "Attendance %",
        **SUBJECT_LABELS,
    })
    return out.reset_index(drop=True)


# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', sans-serif !important;
}
#title-block h1 {
    margin-bottom: 2px;
    font-size: 26px;
}
#title-block p {
    margin-top: 0;
    color: #6b7280;
    font-size: 14px;
}
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 12px 0 20px 0;
}
.kpi-card {
    background: #fafbfc;
    border: 1px solid #e5e7eb;
    border-top: 3px solid #1f2937;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.kpi-label {
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    font-weight: 600;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #1f2937;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-sub {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
}
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #1f2937;
    margin: 4px 0 8px 2px;
}
div[role="tablist"] button {
    font-weight: 700 !important;
    font-size: 15px !important;
}
div[role="tablist"] button[aria-selected="true"] {
    color: #1f2937 !important;
    border-bottom: 3px solid #3f5d8a !important;
}
@media (max-width: 900px) {
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
"""

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

with gr.Blocks(title="Student Performance Dashboard") as demo:

    gr.HTML(
        """
        <div id="title-block">
            <h1>Student Performance Dashboard</h1>
            <p>Day 7 mini project - MLB Internship</p>
        </div>
        """
    )

    gr.HTML(build_kpi_html(df))

    with gr.Tabs():

        with gr.Tab("Overview"):
            gr.HTML("<div class='section-title'>Top 5 performers</div>")
            gr.Dataframe(value=get_top5(df), interactive=False, wrap=True)

            gr.HTML("<div class='section-title'>Students needing improvement</div>")
            gr.Dataframe(value=get_needs_improvement(df), interactive=False, wrap=True)

            gr.HTML("<div class='section-title'>Average score per subject</div>")
            gr.Plot(value=chart_subject_averages(df))

            gr.HTML("<div class='section-title'>Distribution of average scores</div>")
            gr.Plot(value=chart_score_distribution(df))

        with gr.Tab("Charts"):
            gr.HTML("<div class='section-title'>Python score vs Machine Learning score</div>")
            gr.Plot(value=chart_python_vs_ml(df))

            gr.HTML("<div class='section-title'>Score spread across subjects</div>")
            gr.Plot(value=chart_subject_spread(df))

            gr.HTML("<div class='section-title'>Students by performance category</div>")
            gr.Plot(value=chart_performance_breakdown(df))

            gr.HTML("<div class='section-title'>Performance category share</div>")
            gr.Plot(value=chart_performance_pie(df))

        with gr.Tab("Full Dataset"):
            gr.HTML("<div class='section-title'>Complete student records</div>")

            sort_filter = gr.Dropdown(
                choices=SORT_OPTIONS,
                value="Average Score",
                label="Sort by",
            )

            full_table = gr.Dataframe(
                value=get_full_dataset("All", "Average Score"),
                interactive=False,
                wrap=True,
            )

            sort_filter.change(
                fn=lambda sort_by: get_full_dataset("All", sort_by),
                inputs=sort_filter,
                outputs=full_table,
            )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)