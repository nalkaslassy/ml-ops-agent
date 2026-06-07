"""
Tests every behaviour the user specified:
  1. Smoke tests  — all 4 models train + predict end-to-end
  2. Validation   — all incompatible-data scenarios caught before training
  3. Router       — plain-English description maps to the right model
  4. Results      — results_ dict stored, best_score stored, test_score stored
  5. No charts    — no matplotlib import in any model file
  6. UI flows     — browser-driven Selenium tests of the full Train tab flow
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

RESULTS = []
SECTION = ""

def log(name, ok, detail=""):
    RESULTS.append((SECTION, name, ok))
    mark = "[OK]  " if ok else "[FAIL]"
    print(f"  {mark} {name}" + (f"\n         {detail}" if detail else ""))

def section(name):
    global SECTION
    SECTION = name
    print(f"\n=== {name} ===")

# ── Shared data generators ────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def make_churn(n=300, bad_target=False, single_class=False):
    df = pd.DataFrame({
        "customer_id": range(n),
        "tenure":      rng.integers(1, 72, n),
        "spend":       rng.normal(70, 20, n).clip(15, 250).round(2),
        "tickets":     rng.poisson(2, n),
        "contract":    rng.choice(["monthly", "annual"], n),
        "churned":     rng.choice([0,1,2], n) if bad_target else
                       (0 if single_class else (rng.random(n) < 0.15).astype(int)),
    })
    return df

def make_fraud(n=300):
    fraud = np.zeros(n, dtype=int); fraud[:10] = 1; rng.shuffle(fraud)
    return pd.DataFrame({
        "txn_id": range(n), "amount": rng.lognormal(3.7,.8,n).round(2),
        "hour": rng.integers(0,24,n), "is_fraud": fraud,
    })

def make_sentiment(n=300):
    pos = ["Great product", "Love it", "Excellent"]
    neg = ["Terrible", "Broke fast", "Waste of money"]
    labels = rng.choice([0,1], size=n)
    texts  = [rng.choice(pos) if l else rng.choice(neg) for l in labels]
    return pd.DataFrame({"review": texts, "sentiment": labels})

def make_segmentation(n=240):
    arr = np.vstack([
        rng.normal([25,30,20],[4,6,5],(n//3,3)),
        rng.normal([45,65,55],[5,8,7],(n//3,3)),
        rng.normal([60,20,80],[6,5,6],(n-2*(n//3),3)),
    ])
    return pd.DataFrame({"customer_id": range(n),
        "spend": arr[:,0].round(2), "engagement": arr[:,1].round(2), "ltv": arr[:,2].round(2)})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SMOKE TESTS — all 4 models train and predict
# ═══════════════════════════════════════════════════════════════════════════════
section("Smoke tests — end-to-end train + predict")

from churn.churn_model             import ChurnModel
from fraud.fraud_model             import FraudModel
from sentiment.sentiment_model     import SentimentModel
from segmentation.segmentation_model import SegmentationModel

for name, model_cls, df, init_kw, pred_input in [
    ("ChurnModel",       ChurnModel,       make_churn(),       {"target_col":"churned","drop_cols":["customer_id"],"n_iter":1,"cv":2}, make_churn().head(20)),
    ("FraudModel",       FraudModel,       make_fraud(),       {"target_col":"is_fraud","drop_cols":["txn_id"],"n_iter":1,"cv":2},    make_fraud().head(20)),
    ("SentimentModel",   SentimentModel,   make_sentiment(),   {"text_col":"review","target_col":"sentiment","n_iter":1,"cv":2},       make_sentiment()["review"].head(20)),
    ("SegmentationModel",SegmentationModel,make_segmentation(),{"drop_cols":["customer_id"]},                                          make_segmentation().head(20)),
]:
    try:
        m = model_cls(**init_kw)
        m.fit(df)
        preds = m.predict(pred_input)
        log(f"{name} trains and predicts", True, f"shape={preds.shape}, cols={list(preds.columns)}")
    except Exception as e:
        log(f"{name} trains and predicts", False, str(e)[:120])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RESULTS STORED — best_score, results_, test_score after fit
# ═══════════════════════════════════════════════════════════════════════════════
section("Results stored on model after fit")

m = ChurnModel(target_col="churned", drop_cols=["customer_id"], n_iter=1, cv=2)
m.fit(make_churn())
log("best_name set",   bool(m.best_name),   detail=m.best_name)
log("best_score set",  m.best_score is not None, detail=str(round(m.best_score,4)))
log("test_score set",  hasattr(m,"test_score") and m.test_score is not None)
log("results_ dict has all algorithms", len(m.results_) == 3, detail=str(m.results_))

m2 = SegmentationModel(drop_cols=["customer_id"])
m2.fit(make_segmentation())
log("SegmentationModel results_ populated", len(m2.results_) > 0, detail=f"{len(m2.results_)} configs tested")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALIDATION — bad data caught before training
# ═══════════════════════════════════════════════════════════════════════════════
section("Validation — incompatible data caught before fit()")

def expect_error(fn, fragment):
    try:
        fn(); return False, "no error raised"
    except ValueError as e:
        msg = str(e)
        return fragment.lower() in msg.lower(), msg[:100]
    except Exception as e:
        return False, f"wrong exception: {type(e).__name__}: {str(e)[:80]}"

checks = [
    ("Churn — too few rows",         lambda: ChurnModel(target_col="churned").fit(make_churn(50)),           "at least 200"),
    ("Churn — target not found",     lambda: ChurnModel(target_col="nonexistent").fit(make_churn()),         "not found"),
    ("Churn — multiclass target",    lambda: ChurnModel(target_col="churned").fit(make_churn(bad_target=True)), "must be binary"),
    ("Churn — single class target",  lambda: ChurnModel(target_col="churned").fit(make_churn(single_class=True)), "only one class"),
    ("Fraud — no fraud cases",       lambda: FraudModel(target_col="is_fraud").fit(make_fraud().assign(is_fraud=0)), "no fraud cases"),
    ("Fraud — no legit cases",       lambda: FraudModel(target_col="is_fraud").fit(make_fraud().assign(is_fraud=1)), "no legitimate cases"),
    ("Sentiment — too few rows",     lambda: SentimentModel(text_col="review",target_col="sentiment").fit(make_sentiment(50)), "at least 200"),
    ("Sentiment — wrong label count",lambda: SentimentModel(text_col="review",target_col="sentiment").fit(make_sentiment().assign(sentiment=rng.choice([0,1,2],300))), "exactly 2"),
    ("Segmentation — too few rows",  lambda: SegmentationModel(drop_cols=["customer_id"]).fit(make_segmentation(30)), "at least 50"),
    ("Segmentation — no numeric cols",lambda: SegmentationModel(drop_cols=["customer_id","spend","engagement","ltv"]).fit(make_segmentation()), "at least 2 numeric"),
]
for label, fn, fragment in checks:
    ok, detail = expect_error(fn, fragment)
    log(label, ok, detail=detail)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ROUTER — plain-English → correct model
# ═══════════════════════════════════════════════════════════════════════════════
section("Router — plain-English description maps to right model")

from agent.router import route, route_and_get_class

routing_cases = [
    ("Which customers are likely to cancel?",              "ChurnModel"),
    ("Detect suspicious and fraudulent transactions",      "FraudModel"),
    ("Classify these reviews as positive or negative",     "SentimentModel"),
    ("Group customers into natural segments for targeting","SegmentationModel"),
]
for problem, expected in routing_cases:
    result = route(problem, prefer_llm=False)
    ok = result["model"] == expected
    log(f"'{problem[:45]}...' → {expected}", ok, detail=f"got: {result['model']}")

# Router returns reason and data_needed
r = route("predict customer churn", prefer_llm=False)
log("Router returns reason",      bool(r.get("reason")))
log("Router returns data_needed", bool(r.get("data_needed")))

# Validate before training — same as what UI does
try:
    ModelClass = route_and_get_class("predict churn", prefer_llm=False)["model_class"]
    ModelClass(target_col="churned")._validate(make_churn(bad_target=True))
    log("UI pre-check catches incompatible data", False, "no error raised")
except ValueError as e:
    log("UI pre-check catches incompatible data", True, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NO CHARTS — no matplotlib in model files
# ═══════════════════════════════════════════════════════════════════════════════
section("No charts — matplotlib removed from model files")

ROOT = Path(__file__).parent
model_files = [
    ROOT / "churn"        / "churn_model.py",
    ROOT / "fraud"        / "fraud_model.py",
    ROOT / "sentiment"    / "sentiment_model.py",
    ROOT / "segmentation" / "segmentation_model.py",
]
for f in model_files:
    content = f.read_text()
    has_mpl = "import matplotlib" in content or "plt.show" in content or "_plot" in content
    log(f"{f.name} — no matplotlib / _plot", not has_mpl)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. UI — Selenium browser tests
# ═══════════════════════════════════════════════════════════════════════════════
section("UI — browser-driven tests")

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

DATA      = Path(__file__).parent / "test_data"
DATA.mkdir(exist_ok=True)
CHURN_CSV = str(DATA / "churn_test.csv")
MULTI_CSV = str(DATA / "multiclass_test.csv")
SENT_CSV  = str(DATA / "sentiment_test.csv")
SHOTS     = DATA

# Generate test CSVs if they don't already exist
if not (DATA / "churn_test.csv").exists():
    _n = 300
    _t = rng.integers(1, 72, _n); _s = rng.normal(70,20,_n).clip(15,250).round(2)
    _c = rng.choice(["monthly","annual"],_n); _churned = (rng.random(_n)<0.15).astype(int)
    pd.DataFrame({"customer_id":range(_n),"tenure_months":_t,"monthly_spend":_s,
        "support_tickets":rng.poisson(2,_n),"contract_type":_c,"churned":_churned}
    ).to_csv(CHURN_CSV, index=False)
    mc = pd.read_csv(CHURN_CSV)
    mc["churned"] = rng.choice([0,1,2], _n)
    mc.to_csv(MULTI_CSV, index=False)
    _labels = rng.choice([0,1],size=_n,p=[0.48,0.52])
    _pos=["Great product","Love it","Excellent"]; _neg=["Terrible","Broke fast","Waste of money"]
    _texts=[rng.choice(_pos) if l else rng.choice(_neg) for l in _labels]
    pd.DataFrame({"review":_texts,"sentiment":_labels}).to_csv(SENT_CSV, index=False)

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--window-size=1400,1000")
opts.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

driver = webdriver.Chrome(options=opts)
driver.implicitly_wait(5)

def wait(s=1.5): time.sleep(s)
def body(): return driver.find_element(By.TAG_NAME, "body").text
def get_alerts(): return [e.text for e in driver.find_elements(By.CSS_SELECTOR, '[data-testid="stAlert"]')]
def fresh(): driver.get("http://localhost:8501"); wait(3)
def click_tab(name):
    for t in driver.find_elements(By.CSS_SELECTOR, 'button[role="tab"]'):
        if name.lower() in t.text.lower(): t.click(); wait(1.5); return True
    return False
def fill_problem(text):
    for inp in driver.find_elements(By.CSS_SELECTOR, "input[type=text]"):
        if inp.is_displayed(): inp.click(); inp.clear(); inp.send_keys(text); inp.send_keys(Keys.RETURN); wait(2); return True
    return False
def upload_csv(path):
    for inp in driver.find_elements(By.CSS_SELECTOR, "input[type=file]"):
        try: inp.send_keys(path); wait(2.5); return True
        except Exception: pass
    return False
def pick_option(text, idx=0):
    dds = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb=select]")
    if not dds or idx >= len(dds): return False
    dds[idx].click(); wait(0.5)
    for opt in driver.find_elements(By.CSS_SELECTOR, "[role=option]"):
        if text.lower() in opt.text.lower(): opt.click(); wait(2); return True
    return False

try:
    # ── 6a. Page structure ────────────────────────────────────────────────────
    fresh()
    bt = body()
    log("Page loads with title", "ML Ops Agent" in bt)
    log("Train / Predict / About tabs present",
        all(t in bt for t in ["Train", "Predict", "About"]))
    log("Supported models table visible on Train page",
        "Churn Prediction" in bt and "Fraud Detection" in bt and "Sentiment Analysis" in bt)
    log("Data format requirements visible on Train page",
        "CSV" in bt and "cleaned" in bt.lower() or "first row" in bt.lower())
    driver.save_screenshot(str(SHOTS / "ui_01_landing.png"))

    # ── 6b. Router + compatibility green ─────────────────────────────────────
    fresh()
    fill_problem("Which customers are likely to cancel?")
    bt = body()
    log("Router picks ChurnModel and shows reason", "ChurnModel" in bt)

    upload_csv(CHURN_CSV)
    driver.execute_script("window.scrollTo(0,400)")
    pick_option("churned", idx=0)
    alrt = get_alerts()
    alrt_text = " | ".join(alrt)
    log("Compatibility check green for valid data",
        any("compatible" in a.lower() and "not compatible" not in a.lower() for a in alrt),
        detail=alrt_text[:120])
    driver.save_screenshot(str(SHOTS / "ui_02_compat_green.png"))

    # ── 6c. Incompatible data — error + alternatives shown ───────────────────
    fresh()
    fill_problem("Which customers are likely to cancel?")
    upload_csv(MULTI_CSV)
    driver.execute_script("window.scrollTo(0,400)")
    pick_option("churned", idx=0)
    alrt = get_alerts()
    alrt_text = " | ".join(alrt)
    log("Red error shown for incompatible data",
        any("not compatible" in a.lower() for a in alrt),
        detail=alrt_text[:120])
    log("Compatible alternatives listed",
        any(m in alrt_text for m in ["SentimentModel", "SegmentationModel"]),
        detail=alrt_text[:150])
    bt = body()
    log("'Use X instead' buttons present",
        any("Use " in line for line in bt.splitlines()),
        detail=next((l for l in bt.splitlines() if "Use " in l), "not found")[:80])
    driver.save_screenshot(str(SHOTS / "ui_03_incompatible.png"))

    # ── 6d. About tab ─────────────────────────────────────────────────────────
    click_tab("About")
    bt = body()
    log("About tab: all 4 models described",
        all(m in bt for m in ["Churn", "Fraud", "Sentiment", "Segmentation"]))
    log("About tab: algorithm table present", "ROC-AUC" in bt)
    driver.save_screenshot(str(SHOTS / "ui_04_about.png"))

    # ── 6e. Full train flow (fast settings) ───────────────────────────────────
    fresh()
    fill_problem("Which customers are likely to cancel?")
    upload_csv(CHURN_CSV)
    driver.execute_script("window.scrollTo(0,400)")
    pick_option("churned", idx=0)

    # Open Advanced settings and minimise sliders
    try:
        adv = driver.find_elements(By.XPATH, '//*[contains(text(),"Advanced settings")]')
        if adv: adv[0].click(); wait(1)
        for sl in driver.find_elements(By.CSS_SELECTOR, 'input[type="range"]'):
            mn = sl.get_attribute("min") or "1"
            driver.execute_script(
                "arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}))",
                sl, mn)
        wait(1)
    except Exception: pass

    # Click Train
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        if "train model" in btn.text.lower(): btn.click(); break

    print("  Waiting for training (up to 3 min)...")
    deadline = time.time() + 180
    trained  = False
    while time.time() < deadline:
        wait(5)
        bt = body()
        if "training complete" in bt.lower() or "how every algorithm compared" in bt.lower():
            trained = True; break

    driver.save_screenshot(str(SHOTS / "ui_05_after_train.png"))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    wait(1)
    driver.save_screenshot(str(SHOTS / "ui_06_results.png"))

    bt = body()
    log("Training completes successfully", trained)
    log("Algorithm comparison table shown", "how every algorithm compared" in bt.lower() or "logistic regression" in bt.lower())
    log("'Why this algorithm was chosen' section shown", "why this algorithm" in bt.lower())
    log("Download model button shown", "download trained model" in bt.lower())
    # Old chart titles would appear if matplotlib plots were still rendered
    old_chart_titles = ["roc curve", "confusion matrix", "precision-recall curve"]
    no_old_charts = not any(t in bt.lower() for t in old_chart_titles)
    log("No matplotlib chart titles in results (text only)", no_old_charts,
        detail="found old chart title" if not no_old_charts else "clean")

finally:
    driver.quit()


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
by_section = {}
for sec, name, ok in RESULTS:
    by_section.setdefault(sec, []).append(ok)

total_pass = sum(ok for _, _, ok in RESULTS)
total      = len(RESULTS)

for sec, okays in by_section.items():
    p = sum(okays); t = len(okays)
    marker = "OK" if p == t else "FAIL"
    print(f"  [{marker}] {sec}: {p}/{t}")

print()
print(f"TOTAL: {total_pass}/{total} passed")
print("OVERALL:", "PASS" if total_pass == total else "FAIL")
if total_pass < total:
    print("\nFailed:")
    for sec, name, ok in RESULTS:
        if not ok: print(f"  - [{sec}] {name}")
