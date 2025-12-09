#!/usr/bin/env python3
"""Create PowerPoint presentation for Connecticut Housing Price Forecasting"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
import os

# Create presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle):
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(12.333), Inches(1))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(24)
    p.alignment = PP_ALIGN.CENTER

    return slide

def add_content_slide(prs, title, bullets, image_path=None):
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True

    # Content area depends on whether there's an image
    if image_path and os.path.exists(image_path):
        # Bullets on left, image on right
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(5.5), Inches(5.5))
        tf = content_box.text_frame
        tf.word_wrap = True

        for i, bullet in enumerate(bullets):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = "• " + bullet
            p.font.size = Pt(20)
            p.space_after = Pt(12)

        # Add image
        slide.shapes.add_picture(image_path, Inches(6.5), Inches(1.3), width=Inches(6.3))
    else:
        # Full width bullets
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12.333), Inches(5.5))
        tf = content_box.text_frame
        tf.word_wrap = True

        for i, bullet in enumerate(bullets):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = "• " + bullet
            p.font.size = Pt(24)
            p.space_after = Pt(14)

    return slide

def add_table_slide(prs, title, headers, rows):
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True

    # Table
    num_rows = len(rows) + 1
    num_cols = len(headers)

    table_width = Inches(12)
    table_height = Inches(0.4 * num_rows)
    left = Inches(0.667)
    top = Inches(1.5)

    table = slide.shapes.add_table(num_rows, num_cols, left, top, table_width, table_height).table

    # Set column widths
    col_width = table_width / num_cols
    for i in range(num_cols):
        table.columns[i].width = int(col_width)

    # Header row
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(16)

    # Data rows
    for row_idx, row in enumerate(rows):
        for col_idx, cell_text in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(cell_text)
            cell.text_frame.paragraphs[0].font.size = Pt(14)

    return slide

def add_image_slide(prs, title, image_path, caption=None):
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True

    # Image centered
    if os.path.exists(image_path):
        slide.shapes.add_picture(image_path, Inches(1.5), Inches(1.2), width=Inches(10))

    # Caption
    if caption:
        cap_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.8), Inches(12.333), Inches(0.5))
        tf = cap_box.text_frame
        p = tf.paragraphs[0]
        p.text = caption
        p.font.size = Pt(16)
        p.font.italic = True
        p.alignment = PP_ALIGN.CENTER

    return slide

# ============================================================
# BUILD THE PRESENTATION
# ============================================================

# Slide 1: Title
add_title_slide(prs,
    "Connecticut Housing Price Forecasting",
    "A Comparative Analysis of Time Series and Regression Models\n\nGMGT-643 Data Analytics\nDecember 2025")

# Slide 2: The Problem
add_content_slide(prs, "Why Forecast Housing Prices?", [
    "Housing affects everyone: buyers, investors, lenders",
    "Better predictions = better decisions",
    "Research Question: Which forecasting model works best?",
    "",
    "Why Connecticut?",
    "Largest public real estate dataset in the country (1.1M+ transactions)",
    "Interesting market: NYC proximity, COVID-19 migration impact"
])

# Slide 3: Data Sources
add_table_slide(prs, "Data Sources",
    ["Source", "Data", "Details"],
    [
        ["CT Open Data Portal", "Housing Transactions", "1.1M records (2001-2024)"],
        ["FRED Database", "Economic Indicators", "Mortgage, Unemployment, CPI, Population"]
    ])

# Slide 4: Data Preparation
add_content_slide(prs, "Data Preparation", [
    "Filtered transactions under $10,000 (arm's length only)",
    "Residential properties only (no commercial)",
    "Created monthly MEDIAN prices (not average - avoids outlier skew)",
    "",
    "Final Dataset:",
    "279 months of data (1999-2024)",
    "70/30 train-test split (195 / 84 months)",
    "Test period includes COVID-19 pandemic (stress test)"
])

# Slide 5: Price Trends
add_image_slide(prs, "Connecticut Housing Price Trends",
    "charts/01_price_trend.png",
    "Key events: 2008 bubble/crash, long recovery, COVID-19 surge (2020+)")

# Slide 6: Correlations
add_image_slide(prs, "Economic Indicator Correlations",
    "charts/03_correlation_matrix.png",
    "CPI and population strongly correlated; mortgage rates show complex time-lag relationship")

# Slide 7: Seasonal Decomposition
add_image_slide(prs, "Seasonal Decomposition",
    "charts/04_seasonal_decomposition.png",
    "Clear upward trend, seasonal peaks in spring/summer, COVID surge visible in 2020")

# Slide 8: Models Overview
add_table_slide(prs, "9 Forecasting Models Tested",
    ["Model", "Type", "Key Feature"],
    [
        ["Naive", "Baseline", "Last value = future"],
        ["Seasonal Naive", "Baseline", "Same month last year"],
        ["Moving Average", "Time Series", "3-month window (tuned)"],
        ["Holt-Winters", "Time Series", "Trend + seasonality"],
        ["Ridge (Tuned)", "Regression", "Lagged features"],
        ["XGBoost", "Machine Learning", "Non-linear patterns"],
        ["Prophet", "Time Series", "Facebook's tool"],
        ["SARIMAX", "Time Series", "ARIMA + economic data"],
        ["Weighted Ensemble", "Combined", "Best models averaged"]
    ])

# Slide 9: Feature Engineering
add_table_slide(prs, "Key Insight: Feature Engineering",
    ["Ridge Configuration", "MAPE", "Improvement"],
    [
        ["Current indicators only", "~14%", "Baseline"],
        ["With lagged prices (1,3,6,12 months)", "3.35%", "10+ percentage points!"]
    ])

# Add explanation slide for feature engineering
add_content_slide(prs, "Why Lagged Features Work", [
    "Housing prices have momentum",
    "If prices went up last month, likely to go up this month",
    "Economic changes take time to affect prices",
    "Past prices are highly predictive of future prices",
    "",
    "This was the biggest lesson:",
    "Feature engineering matters more than model complexity"
])

# Slide 10: Results
add_image_slide(prs, "Model Performance Results",
    "charts/05_model_comparison.png",
    "Ridge Regression (3.35% MAPE) outperformed all other models")

# Slide 11: Results Table
add_table_slide(prs, "Model Performance Comparison",
    ["Model", "MAPE", "Interpretation"],
    [
        ["Ridge (Tuned)", "3.35%", "$100K home → off by $3,350"],
        ["Weighted Ensemble", "3.83%", "Combined best models"],
        ["Moving Average", "5.08%", "Simple but effective"],
        ["Holt-Winters", "5.20%", "Captures seasonality"],
        ["Seasonal Naive", "6.81%", "Good baseline"],
        ["XGBoost", "11.21%", "Machine learning approach"],
        ["Naive", "19.27%", "Poor baseline"],
        ["Prophet", "22.21%", "Underperformed"]
    ])

# Slide 12: Forecast vs Actual
add_image_slide(prs, "Forecasts vs Actual Prices",
    "charts/06_forecast_vs_actual.png",
    "Top models tracked actual prices closely, even during COVID-19 surge")

# Slide 13: Limitations
add_content_slide(prs, "Limitations", [
    "State-level data only",
    "   - Local markets vary (Fairfield County vs rural CT)",
    "",
    "Lagged features assume trends continue",
    "   - May be slow to detect market turning points",
    "",
    "Results specific to this time period",
    "   - Performance may differ in other market conditions"
])

# Slide 14: Key Takeaways
add_content_slide(prs, "Key Takeaways", [
    "Feature engineering > model complexity",
    "   - Lagged prices improved MAPE by 10+ percentage points",
    "",
    "Simple models work when tuned",
    "   - Moving Average and Holt-Winters achieved ~5% MAPE",
    "",
    "Baselines are essential",
    "   - Without Naive model, can't measure improvement",
    "",
    "Best Model: Ridge Regression with Lagged Features (3.35% MAPE)"
])

# Slide 15: Questions
add_title_slide(prs, "Questions?", "Thank you!\n\nData: Connecticut Open Data Portal\nEconomic Data: FRED (Federal Reserve)")

# Save presentation
prs.save('Connecticut_Housing_Forecasting_Presentation.pptx')
print("Presentation saved: Connecticut_Housing_Forecasting_Presentation.pptx")
print(f"Total slides: {len(prs.slides)}")
