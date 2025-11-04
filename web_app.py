# web_app.py
import os, json, io
import numpy as np
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ga_feature_selection import GAFeatureSelector, GAConfig, build_estimator

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>GA Feature Selection</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>اختيار الميزات — خوارزمية جينية</h1>
            <p class="subtitle">ارفع ملف CSV، حدِّد عمود الهدف، واضغط تشغيل لمشاهدة النتائج.</p>
        </header>

        {% if error %}
        <div class="error-card">{{ error }}</div>
        {% endif %}

        <form method="post" enctype="multipart/form-data" class="form-card">
            <div class="form-row">
                <label>ملف CSV:</label>
                <input type="file" name="file" accept=".csv">
            </div>

            <div class="form-row">
                <label>اسم عمود الهدف:</label>
                <input type="text" name="target" value="target">
            </div>

            <div class="form-row">
                <label>النموذج:</label>
                <select name="model">
                    <option value="logreg">Logistic Regression</option>
                    <option value="rf">Random Forest</option>
                </select>
            </div>

            <div class="form-row grid-2">
                <label>حجم السكان:</label>
                <input type="number" name="pop" value=60>

                <label>عدد الأجيال:</label>
                <input type="number" name="gens" value=30>
            </div>

            <div class="form-row grid-2">
                <label>Alpha:</label>
                <input type="text" name="alpha" value="1.0">

                <label>Beta:</label>
                <input type="text" name="beta" value="0.4">
            </div>

            <div class="form-row actions">
                <input type="submit" value="تشغيل" class="btn-primary">
            </div>
        </form>

        {% if result %}
        <section class="result-card">
            <h2>النتائج</h2>
            <div class="result-row"><span class="label">الدقّة (CV):</span><span class="value">{{ result.cv_accuracy|round(3) }}</span></div>
            <div class="result-row"><span class="label">نسبة الميزات المختارة:</span><span class="value">{{ result.selected_ratio|round(3) }}</span></div>
            <div class="result-row"><span class="label">عدد الميزات المختارة:</span><span class="value">{{ result.selected_indices|length }}</span></div>
            <div class="result-row"><span class="label">الميزات المختارة:</span><span class="value">{{ selected_features }}</span></div>
        </section>
        {% endif %}

        <footer class="footer-note">النتائج تحفظ تلقائياً إلى <code>results/web_last_result.json</code></footer>
    </div>
</body>
</html>
"""

def prepare_xy(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    # Ensure there is at least one feature column
    if X.shape[1] == 0:
        raise ValueError("لا توجد ميزات لاختبارها: يجب أن يحتوي ملف CSV على عمود/أعمدة ميزات بالإضافة إلى عمود الهدف.")
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X, y, df.columns.drop(target_col)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    selected_features = None
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        target = request.form.get("target", "target")
        model = request.form.get("model", "logreg")
        pop = int(request.form.get("pop", 60))
        gens = int(request.form.get("gens", 30))
        alpha = float(request.form.get("alpha", 1.0))
        beta = float(request.form.get("beta", 0.4))

        if not file:
            error = "لم يتم رفع ملف. يرجى اختيار ملف CSV ثم المحاولة مرة أخرى."
            return render_template_string(HTML, result=None, selected_features=None, error=error)

        try:
            df = pd.read_csv(io.BytesIO(file.read()))
            X, y, feature_names = prepare_xy(df, target)
            est = build_estimator(model)
            cfg = GAConfig(population_size=pop, n_generations=gens, alpha=alpha, beta=beta)

            ga = GAFeatureSelector(est, cfg)
            res = ga.fit(X, y)
        except Exception as e:
            # Show a friendly error message on the page instead of a full traceback
            error = str(e)
            return render_template_string(HTML, result=None, selected_features=None, error=error)

        # Map indices to names
        selected_features = [feature_names[i] for i in res["selected_indices"]]

        # Save JSON
        out_path = os.path.join("results", "web_last_result.json")

        serializable = {
            "selected_features": selected_features,
            "selected_ratio": float(res["selected_ratio"]),
            "cv_accuracy": float(res["cv_accuracy"]),
            "config": make_json_serializable(vars(res["config"])),
            "history": make_json_serializable(res["history"]),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        # Convert dict to object for template access
        class R: pass
        r = R()
        r.cv_accuracy = res["cv_accuracy"]
        r.selected_ratio = res["selected_ratio"]
        r.selected_indices = res["selected_indices"]
        result = r

    return render_template_string(HTML, result=result, selected_features=selected_features, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
