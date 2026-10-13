"""
Data Cleaning & Reporting Automation
Usage:
    python clean_and_report.py                   -> generates messy sample data, cleans it, builds reports
    python clean_and_report.py my_data.csv       -> same pipeline on your own CSV / XLSX
Outputs (in ./reports/<timestamp>/): cleaned_data.csv, report.xlsx, report.html, charts/*.png, cleaning_log.csv
Install: pip install pandas numpy matplotlib openpyxl
Schedule daily on Windows:  schtasks /create /sc daily /st 09:00 /tn "AutoReport" /tr "python C:\\path\\clean_and_report.py data.csv"
"""
import sys, os, re, base64, difflib, io
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join("reports", STAMP); os.makedirs(f"{OUT}/charts", exist_ok=True)
LOG = []
def log(step, detail, n=None): LOG.append({"step": step, "detail": detail, "rows_affected": n}); print(f"[{step}] {detail}" + (f" -> {n}" if n is not None else ""))

# ---------- 0. SAMPLE MESSY DATA ----------
def messy_sample(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    cities = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai"]
    cats = ["Electronics", "Clothing", "Grocery", "Furniture"]
    d = pd.DataFrame({
        "OrderID": range(1000, 1000 + n),
        "Customer": [f"customer {rng.integers(1, 300)}" for _ in range(n)],
        "Email": [f"user{rng.integers(1,300)}@mail.com" for _ in range(n)],
        "City": rng.choice(cities, n), "Category": rng.choice(cats, n),
        "Quantity": rng.integers(1, 10, n), "UnitPrice": rng.integers(100, 5000, n).astype(float),
        "OrderDate": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "D"),
    })
    d["OrderDate"] = [x.strftime(rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"])) for x in d.OrderDate]
    d["City"] = [rng.choice([c, c.upper(), c.lower(), " " + c + " ", c[:-1]]) if rng.random() < .25 else c for c in d.City]   # inconsistent
    d["Category"] = [rng.choice([c.lower(), c.upper(), c + " "]) if rng.random() < .2 else c for c in d.Category]
    d["UnitPrice"] = d.UnitPrice.astype(object); idx = rng.choice(n, 80, replace=False)
    d.loc[idx, "UnitPrice"] = ["$" + str(int(float(v))) + ",00".replace(",00", "") for v in d.loc[idx, "UnitPrice"]]
    for c, k in [("Customer", 60), ("City", 50), ("Quantity", 70), ("UnitPrice", 70), ("Email", 40), ("OrderDate", 30)]:
        d.loc[rng.choice(n, k, replace=False), c] = np.nan
    d.loc[rng.choice(n, 12, replace=False), "Quantity"] = -5                       # invalid
    d.loc[rng.choice(n, 10, replace=False), "UnitPrice"] = 900000                 # outliers
    d.loc[rng.choice(n, 25, replace=False), "Email"] = "not_an_email"
    d = pd.concat([d, d.sample(45, random_state=1)], ignore_index=True)           # duplicates
    return d.sample(frac=1, random_state=2).reset_index(drop=True)

# ---------- 1. LOAD ----------
if len(sys.argv) > 1:
    p = sys.argv[1]; raw = pd.read_excel(p) if p.lower().endswith(("xlsx", "xls")) else pd.read_csv(p)
else:
    raw = messy_sample(); raw.to_csv("sample_messy_data.csv", index=False)
df = raw.copy()
log("load", f"{df.shape[0]} rows x {df.shape[1]} cols")
before_missing = int(df.isna().sum().sum())

# ---------- 2. CLEAN ----------
df.columns = [re.sub(r"\W+", "_", c.strip()).strip("_") for c in df.columns]                      # tidy headers
obj = df.select_dtypes(include=["object", "string"]).columns
for c in obj: df[c] = df[c].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA})
log("whitespace", "trimmed text columns & normalised null tokens")

n0 = len(df); df = df.drop_duplicates(); log("duplicates", "exact duplicate rows removed", n0 - len(df))
if "OrderID" in df: n0 = len(df); df = df.drop_duplicates("OrderID"); log("duplicates", "duplicate OrderIDs removed", n0 - len(df))

# numeric columns stored as text ($, commas)
for c in df.columns:
    if df[c].dtype == "string":
        t = pd.to_numeric(df[c].str.replace(r"[$,₹\s]", "", regex=True), errors="coerce")
        if t.notna().mean() > .8: df[c] = t; log("numeric", f"converted '{c}' text -> number")

# dates in mixed formats
for c in df.columns:
    if "date" in c.lower():
        parsed = pd.to_datetime(df[c], format="mixed", dayfirst=True, errors="coerce"); df[c] = parsed
        log("dates", f"parsed '{c}' (mixed formats)", int(parsed.notna().sum()))

# categorical standardisation + fuzzy typo fix
for c in df.select_dtypes("string").columns:
    if df[c].nunique() < 30 and "email" not in c.lower():
        df[c] = df[c].str.title()
        counts = df[c].value_counts(); canon = list(counts[counts > counts.max() * .1].index)
        def fix(v):
            if pd.isna(v) or v in canon: return v
            m = difflib.get_close_matches(v, canon, 1, .8); return m[0] if m else v
        changed = (df[c] != df[c].map(fix)).sum(); df[c] = df[c].map(fix)
        log("standardise", f"'{c}' case/typos unified", int(changed))

# email validation
if "Email" in df:
    ok = df.Email.str.match(r"^[\w.+-]+@[\w-]+\.[\w.]+$", na=False); bad = (~ok & df.Email.notna()).sum()
    df.loc[~ok, "Email"] = pd.NA; log("validate", "invalid emails set to missing", int(bad))

# impossible values
for c in [x for x in df.select_dtypes("number").columns if re.search("qty|quantity|price|amount", x, re.I)]:
    bad = (df[c] <= 0).sum(); df.loc[df[c] <= 0, c] = np.nan; log("validate", f"non-positive '{c}' -> missing", int(bad))

# outliers (IQR cap)
for c in [x for x in df.select_dtypes("number").columns if re.search("price|amount", x, re.I)]:
    q1, q3 = df[c].quantile([.25, .75]); hi = q3 + 3 * (q3 - q1); n = int((df[c] > hi).sum())
    df[c] = df[c].clip(upper=hi); log("outliers", f"'{c}' capped at {hi:.0f}", n)

# missing values: median / mode / label
for c in df.columns:
    m = int(df[c].isna().sum())
    if m == 0 or "id" == c.lower()[-2:]: continue
    if pd.api.types.is_numeric_dtype(df[c]): df[c] = df[c].fillna(df[c].median()); how = "median"
    elif pd.api.types.is_datetime64_any_dtype(df[c]): df[c] = df[c].fillna(df[c].median()); how = "median date"
    elif "email" in c.lower() or "customer" in c.lower(): df[c] = df[c].fillna("Unknown"); how = "'Unknown'"
    else: df[c] = df[c].fillna(df[c].mode()[0]); how = "mode"
    log("missing", f"'{c}' filled with {how}", m)

if {"Quantity", "UnitPrice"} <= set(df.columns): df["Revenue"] = df.Quantity * df.UnitPrice
df = df.reset_index(drop=True)
log("done", f"clean dataset {df.shape[0]} rows | missing {before_missing} -> {int(df.isna().sum().sum())}")

# ---------- 3. SUMMARIES ----------
date_c = next((c for c in df.columns if "date" in c.lower()), None)
sheets = {"Cleaned_Data": df, "Cleaning_Log": pd.DataFrame(LOG), "Describe": df.describe(include="all").T.reset_index()}
kpi = {"Rows (clean)": f"{len(df):,}", "Rows (raw)": f"{len(raw):,}", "Missing fixed": f"{before_missing:,}"}
charts = []
def save(name): plt.tight_layout(); plt.savefig(f"{OUT}/charts/{name}.png", dpi=130); plt.close(); charts.append(name)

if "Revenue" in df:
    kpi["Total revenue"] = f"{df.Revenue.sum():,.0f}"; kpi["Avg order value"] = f"{df.Revenue.mean():,.0f}"
    for dim in [c for c in ["City", "Category"] if c in df]:
        s = df.groupby(dim).Revenue.sum().sort_values(); sheets[f"Revenue_by_{dim}"] = s.sort_values(ascending=False).reset_index()
        s.plot.barh(color="#2563eb", figsize=(7, 4)); plt.title(f"Revenue by {dim}"); save(f"rev_by_{dim}")
    if date_c:
        m = df.set_index(date_c).Revenue.resample("MS").sum(); sheets["Monthly_Revenue"] = m.reset_index()
        m.plot(marker="o", figsize=(9, 4), color="#16a34a"); plt.title("Monthly revenue trend"); save("monthly_revenue")
fig, ax = plt.subplots(figsize=(7, 4)); pd.Series({"Missing before": before_missing, "Missing after": int(df.isna().sum().sum())}).plot.bar(ax=ax, color=["#dc2626", "#16a34a"])
plt.title("Data quality: missing values"); plt.xticks(rotation=0); save("quality")

# ---------- 4. EXPORT ----------
df.to_csv(f"{OUT}/cleaned_data.csv", index=False); pd.DataFrame(LOG).to_csv(f"{OUT}/cleaning_log.csv", index=False)
with pd.ExcelWriter(f"{OUT}/report.xlsx", engine="openpyxl") as w:
    for k, v in sheets.items():
        v.to_excel(w, sheet_name=k[:31], index=False)
        ws = w.sheets[k[:31]]
        for col in ws.columns: ws.column_dimensions[col[0].column_letter].width = min(max(len(str(col[0].value)) + 4, 14), 40)

def img(n): return base64.b64encode(open(f"{OUT}/charts/{n}.png", "rb").read()).decode()
html = f"""<html><head><meta charset='utf-8'><title>Automated Data Report</title><style>
body{{font-family:Segoe UI,Arial;margin:30px;background:#f3f4f6;color:#111}}h1{{color:#1d4ed8}}
.k{{display:flex;gap:14px;flex-wrap:wrap}}.c{{background:#fff;padding:14px 22px;border-radius:10px;box-shadow:0 1px 4px #0002}}.c b{{font-size:22px;display:block;color:#2563eb}}
table{{border-collapse:collapse;background:#fff}}td,th{{border:1px solid #ddd;padding:6px 10px;font-size:13px}}th{{background:#1d4ed8;color:#fff}}
img{{max-width:48%;margin:8px;border-radius:8px;background:#fff}}</style></head><body>
<h1>Automated Data Cleaning &amp; Reporting</h1><p>Generated {datetime.now():%d %b %Y %H:%M}</p>
<div class='k'>{''.join(f"<div class='c'>{k}<b>{v}</b></div>" for k, v in kpi.items())}</div>
<h2>Cleaning log</h2>{pd.DataFrame(LOG).to_html(index=False)}
<h2>Visual summaries</h2>{''.join(f"<img src='data:image/png;base64,{img(n)}'>" for n in charts)}
<h2>Cleaned data preview</h2>{df.head(15).to_html(index=False)}</body></html>"""
open(f"{OUT}/report.html", "w", encoding="utf-8").write(html)
print(f"\nDone. Open {OUT}/report.html")
