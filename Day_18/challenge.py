import os
import glob

import cv2

from mini_project import process_document

INPUT_DIR = "sample_images"
OUTPUT_DIR = "outputs/challenge"
MAX_IMAGES = 10
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def get_image_paths(input_dir, max_images):
    paths = sorted(
        p for p in glob.glob(os.path.join(input_dir, "*"))
        if p.lower().endswith(SUPPORTED_EXT)
    )
    return paths[:max_images]


def expected_outcome(boundary):
    if boundary is None:
        return "No boundary detected — contour not found or image too noisy/low-contrast."
    points = len(boundary)
    if points == 4:
        return "Document boundary detected successfully (clean 4-corner quadrilateral)."
    return f"Boundary detected but with {points} points (not a clean quadrilateral — may need threshold tuning)."


def process_single(image_path, index):
    name = os.path.splitext(os.path.basename(image_path))[0]
    folder = os.path.join(OUTPUT_DIR, f"{index:02d}_{name}")
    os.makedirs(folder, exist_ok=True)

    try:
        results = process_document(image_path, os.path.join(folder, "final.jpg"))
    except ValueError as e:
        return {"name": name, "folder": folder, "error": str(e)}

    cv2.imwrite(os.path.join(folder, "original.jpg"), results["original"])
    cv2.imwrite(os.path.join(folder, "edges.jpg"), results["edges"])
    cv2.imwrite(os.path.join(folder, "morphology.jpg"), results["morphology"])

    return {
        "name": name,
        "folder": folder,
        "outcome": expected_outcome(results["boundary"]),
        "detected": results["boundary"] is not None,
        "error": None,
    }


def generate_html_report(results, report_path, output_dir):
    rows_html = []

    for i, r in enumerate(results, start=1):
        if r.get("error"):
            rows_html.append(
                f"""
                <div class="card">
                    <div class="card-title">{i:02d}. {r['name']}</div>
                    <div class="badge badge-error">❌ Failed to load: {r['error']}</div>
                </div>
                """
            )
            continue

        rel = os.path.relpath(r["folder"], output_dir).replace(os.sep, "/")
        badge_class = "badge-success" if r["detected"] else "badge-warning"
        badge_icon = "✅" if r["detected"] else "⚠️"

        rows_html.append(
            f"""
            <div class="card">
                <div class="card-title">{i:02d}. {r['name']}</div>
                <div class="image-grid">
                    <figure>
                        <img src="{rel}/original.jpg">
                        <figcaption>Original</figcaption>
                    </figure>
                    <figure>
                        <img src="{rel}/edges.jpg">
                        <figcaption>Edge Detection</figcaption>
                    </figure>
                    <figure>
                        <img src="{rel}/morphology.jpg">
                        <figcaption>Morphology</figcaption>
                    </figure>
                    <figure>
                        <img src="{rel}/final.jpg">
                        <figcaption>Final Boundary</figcaption>
                    </figure>
                </div>
                <div class="badge {badge_class}">{badge_icon} {r['outcome']}</div>
            </div>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Document Boundary Detection — Comparison Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #ffffff;
                color: #111827;
                margin: 0;
                padding: 40px;
            }}
            .header {{
                padding: 24px 28px;
                border-radius: 14px;
                background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
                border: 1px solid #e5e9f0;
                margin-bottom: 32px;
            }}
            .header h1 {{ margin: 0; font-size: 26px; }}
            .header p {{ margin: 6px 0 0 0; color: #6b7280; font-size: 14px; }}

            .card {{
                border: 1px solid #e5e9f0;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 24px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }}
            .card-title {{
                font-size: 17px;
                font-weight: 600;
                margin-bottom: 14px;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 14px;
                margin-bottom: 14px;
            }}
            figure {{ margin: 0; text-align: center; }}
            figure img {{
                width: 100%;
                border-radius: 8px;
                border: 1px solid #e5e9f0;
                object-fit: cover;
            }}
            figcaption {{
                margin-top: 6px;
                font-size: 12px;
                color: #6b7280;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 14px;
                border-radius: 999px;
                font-weight: 600;
                font-size: 13px;
            }}
            .badge-success {{ background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
            .badge-warning {{ background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
            .badge-error   {{ background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}

            @media (max-width: 900px) {{
                .image-grid {{ grid-template-columns: repeat(2, 1fr); }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📄 Document Boundary Detection — Comparison Report</h1>
            <p>{len(results)} image(s) processed · Pipeline: Grayscale → Blur → Canny → Morphology → Contour Detection</p>
        </div>
        {''.join(rows_html)}
    </body>
    </html>
    """

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = get_image_paths(INPUT_DIR, MAX_IMAGES)

    if not image_paths:
        print(f"No images found in '{INPUT_DIR}/'. Add up to {MAX_IMAGES} images there and re-run.")
        return

    print(f"Found {len(image_paths)} image(s). Processing...\n")

    all_results = []
    for i, path in enumerate(image_paths, start=1):
        print(f"  [{i}/{len(image_paths)}] {os.path.basename(path)}")
        all_results.append(process_single(path, i))

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.html")
    generate_html_report(all_results, report_path, OUTPUT_DIR)

    print(f"\nDone. Open the report in your browser:\n  {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()