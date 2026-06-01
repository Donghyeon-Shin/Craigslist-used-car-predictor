# =============================================================================
# Used Car Price Prediction — Regression Model Analysis
# Data   : preprocessed_vehicles.csv
# Model  : Linear Regression (Baseline vs Enhanced with estimate_msrp)
# Flow   : Load → Split → log1p → VIF removal → Optimal step → Eval → Plot
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings('ignore')
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'


# -----------------------------------------------------------------------------
# Step 1. Load Data
# -----------------------------------------------------------------------------
df = pd.read_csv('./Data/preprocessed_vehicles.csv') 

print(f'Shape: {df.shape}  |  Missing: {df.isnull().sum().sum()}')
print(df.head(3))

numeric_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'price']
print(f'\nNumeric features for VIF: {len(numeric_feats)}')


# -----------------------------------------------------------------------------
# Step 2. Train / Test Split (8:2)
# -----------------------------------------------------------------------------
all_feats = numeric_feats + ['model']
X_full = df[all_feats].copy()
y      = df['price'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
print(f'Train: {X_train.shape}  Test: {X_test.shape}')


# -----------------------------------------------------------------------------
# Step 3. Target Log Transformation (log1p)
# Used car prices are right-skewed; log1p brings them closer to normal.
# Predictions are restored to dollar units with expm1() afterwards.
# -----------------------------------------------------------------------------
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train,     bins=50, color='#3498DB', edgecolor='white')
axes[0].set_title('Original price Distribution', fontweight='bold')
axes[0].set_xlabel('price ($)')
axes[1].hist(y_train_log, bins=50, color='#E74C3C', edgecolor='white')
axes[1].set_title('log1p(price) Distribution', fontweight='bold')
axes[1].set_xlabel('log1p(price)')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Original range : ${y_train.min():,.0f} ~ ${y_train.max():,.0f}')
print(f'Transformed    : {y_train_log.min():.3f} ~ {y_train_log.max():.3f}')


# -----------------------------------------------------------------------------
# Step 4. Helper Functions
# -----------------------------------------------------------------------------

def encode_and_scale(X_train, X_test, y_train_log,
                     target_col='model',
                     scale_cols=None,
                     smooth=10):
    """
    Target Encoding for `model` column + Standard Scaling for numeric columns.
    Fit only on training data to prevent leakage.
    """
    if scale_cols is None:
        scale_cols = ['odometer', 'vehicle_age']

    X_tr = X_train.copy()
    X_te = X_test.copy()

    te = None
    encoded_col = f'{target_col}_encoded'
    if target_col in X_tr.columns:
        te = TargetEncoder(smooth=smooth, random_state=42)
        X_tr[encoded_col] = te.fit_transform(X_tr[[target_col]], y_train_log).flatten()
        X_te[encoded_col] = te.transform(X_te[[target_col]]).flatten()
        X_tr.drop(columns=[target_col], inplace=True)
        X_te.drop(columns=[target_col], inplace=True)

    cols_to_scale = scale_cols + ([encoded_col] if te else [])
    cols_to_scale = [c for c in cols_to_scale if c in X_tr.columns]
    scaler = StandardScaler()
    X_tr[cols_to_scale] = scaler.fit_transform(X_tr[cols_to_scale])
    X_te[cols_to_scale] = scaler.transform(X_te[cols_to_scale])

    return X_tr, X_te, te, scaler


def adjusted_r2(r2, n, p):
    """
    Adjusted R² — penalises adding features that do not improve the model.
    Used throughout to fairly compare steps with different feature counts.

    r2 : standard R²
    n  : number of test samples
    p  : number of features
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


print('Functions defined.')


# -----------------------------------------------------------------------------
# Step 5. Define OHE Groups
# One-Hot Encoded features from the same original column must be removed
# together during VIF elimination; dropping only one column causes the
# remaining dummies' VIFs to spike immediately.
# -----------------------------------------------------------------------------
OHE_PREFIXES = [
    'cylinders', 'manufacturer', 'fuel', 'transmission',
    'type', 'title_status', 'drive', 'paint_color',
]

OHE_GROUPS = {}
for prefix in OHE_PREFIXES:
    group_feats = [c for c in numeric_feats if c.startswith(prefix + '_')]
    if group_feats:
        OHE_GROUPS[prefix] = group_feats

# Reverse map: feature → group name
FEAT_TO_GROUP = {feat: grp for grp, feats in OHE_GROUPS.items() for feat in feats}

print('OHE groups:')
for grp, feats in OHE_GROUPS.items():
    print(f'  {grp:15s} ({len(feats):2d}): {feats}')


# -----------------------------------------------------------------------------
# Step 6. Iterative VIF Removal + Performance Tracking
# At each step: compute VIF → train & record metrics → drop highest-VIF feature
# (or its full OHE group) → repeat until all VIFs < 10.
# -----------------------------------------------------------------------------

# --- Inner split: Train → inner Train(64%) + Validation(16%) ---
X_tr_inner, X_val_inner, y_tr_inner_log, y_val_inner_log = train_test_split(
    X_train, y_train_log, test_size=0.2, random_state=42
)
print(f'Inner Train: {X_tr_inner.shape}  Validation: {X_val_inner.shape}')

current_numeric = numeric_feats.copy()
history = []

print(f'\nStarting feature count: {len(current_numeric)}')
print('=' * 85)

step = 0
while len(current_numeric) > 1:

    # --- VIF calculation (inner train 기준) ---
    vif_data  = X_tr_inner[current_numeric].copy()
    vif_const = add_constant(vif_data)
    vif_vals  = [
        variance_inflation_factor(vif_const.values, i + 1)
        for i in range(len(current_numeric))
    ]
    max_vif      = max(vif_vals)
    max_vif_feat = current_numeric[int(np.argmax(vif_vals))]

    # --- Train Baseline (no estimate_msrp), evaluate on Validation ---
    feats     = current_numeric + ['model']
    X_tr_iter = X_tr_inner[feats].drop(columns=['estimate_msrp'], errors='ignore')
    X_va_iter = X_val_inner[feats].drop(columns=['estimate_msrp'], errors='ignore')

    sc = [c for c in ['odometer', 'vehicle_age'] if c in X_tr_iter.columns]
    X_tr_enc, X_va_enc, _, _ = encode_and_scale(X_tr_iter, X_va_iter, y_tr_inner_log, scale_cols=sc)

    lr = LinearRegression()
    lr.fit(X_tr_enc, y_tr_inner_log)
    y_pred = np.clip(np.expm1(lr.predict(X_va_enc)), 0, None)
    y_true = np.expm1(y_val_inner_log)

    n_val  = len(y_true)
    p_val  = X_va_enc.shape[1]
    r2     = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, n_val, p_val)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))

    # --- Determine which features to remove ---
    if max_vif_feat in FEAT_TO_GROUP:
        group_name      = FEAT_TO_GROUP[max_vif_feat]
        feats_to_remove = [f for f in OHE_GROUPS[group_name] if f in current_numeric]
        remove_label    = f'[OHE] {group_name} ({len(feats_to_remove)} features)'
    else:
        feats_to_remove = [max_vif_feat]
        remove_label    = max_vif_feat

    history.append({
        'step'         : step,
        'n_features'   : len(current_numeric),
        'max_vif'      : max_vif,
        'max_vif_feat' : max_vif_feat,
        'removed'      : feats_to_remove if max_vif >= 10 else [],
        'remove_label' : remove_label if max_vif >= 10 else '(none)',
        'R²'           : r2,
        'Adj. R²'      : adj_r2,
        'RMSE'         : rmse,
        'features'     : current_numeric.copy(),
    })

    print(f'Step {step:2d} | Features {len(current_numeric):2d} | Max VIF={max_vif:8.2f} '
          f'({max_vif_feat}) | R²={r2:.4f} | Adj.R²={adj_r2:.4f} | RMSE=${rmse:,.0f}')

    if max_vif < 10:
        print('\nAll VIFs < 10 → stopping.')
        break

    if max_vif_feat in FEAT_TO_GROUP:
        print(f'       → Remove OHE group [{FEAT_TO_GROUP[max_vif_feat]}]: {feats_to_remove}')
    else:
        print(f'       → Remove: {max_vif_feat}')

    for f in feats_to_remove:
        if f in current_numeric:
            current_numeric.remove(f)

    step += 1

print('=' * 85)
print(f'{len(history)} steps completed.')


# -----------------------------------------------------------------------------
# Step 7. Select Optimal Feature Combination (highest Adjusted R²)
# The stopping point (VIF < 10) and the best-performing step are separate;
# pick the step with the highest Adjusted R² as the final feature set.
# -----------------------------------------------------------------------------
hist_df  = pd.DataFrame(history)
best_idx = hist_df['Adj. R²'].idxmax()
best_step = hist_df.loc[best_idx]

SELECTED_NUMERIC  = best_step['features']
SELECTED_FEATURES = ['model'] + SELECTED_NUMERIC

print(f'Optimal step  : Step {int(best_step["step"])}')
print(f'Feature count : {len(SELECTED_NUMERIC)} numeric + model')
print(f'Max VIF       : {best_step["max_vif"]:.2f}')
print(f'R²            : {best_step["R²"]:.4f}')
print(f'Adj. R²       : {best_step["Adj. R²"]:.4f}')
print(f'RMSE          : ${best_step["RMSE"]:,.0f}')
print(f'\nSelected features: {SELECTED_FEATURES}')

# --- 4-panel VIF history chart ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
steps      = hist_df['step'].tolist()
n_features = hist_df['n_features'].tolist()
max_vifs   = hist_df['max_vif'].tolist()
r2s        = hist_df['R²'].tolist()
adj_r2s    = hist_df['Adj. R²'].tolist()

axes[0].plot(steps, adj_r2s, marker='o', color='#9B59B6', linewidth=2, label='Adj. R²')
axes[0].axvline(best_idx, color='red', linestyle='--', linewidth=1.5, label=f'Optimal Step {int(best_step["step"])}')
axes[0].scatter([best_idx], [best_step['Adj. R²']], color='red', s=100, zorder=5)
axes[0].set_xlabel('Step'); axes[0].set_ylabel('Adjusted R²')
axes[0].set_title('Adjusted R² Trend (Optimal Step Selection)', fontweight='bold')
axes[0].legend()

axes[1].plot(steps, r2s, marker='o', color='#3498DB', linewidth=2, label='R²')
axes[1].axvline(best_idx, color='red', linestyle='--', linewidth=1.5, label=f'Optimal Step {int(best_step["step"])}')
axes[1].scatter([best_idx], [best_step['R²']], color='red', s=100, zorder=5)
axes[1].set_xlabel('Step'); axes[1].set_ylabel('R²')
axes[1].set_title('R² Trend (Reference)', fontweight='bold')
axes[1].legend()

axes[2].plot(steps, max_vifs, marker='o', color='#E74C3C', linewidth=2)
axes[2].axhline(10, color='red', linestyle='--', linewidth=1.5, label='VIF = 10 threshold')
axes[2].axvline(best_idx, color='blue', linestyle='--', linewidth=1.5, label=f'Optimal Step {int(best_step["step"])}')
axes[2].set_xlabel('Step'); axes[2].set_ylabel('Max VIF')
axes[2].set_title('Max VIF Trend', fontweight='bold')
axes[2].legend()

axes[3].plot(steps, n_features, marker='o', color='#27AE60', linewidth=2)
axes[3].axvline(best_idx, color='red', linestyle='--', linewidth=1.5, label=f'Optimal Step {int(best_step["step"])}')
axes[3].set_xlabel('Step'); axes[3].set_ylabel('Feature Count')
axes[3].set_title('Feature Count Trend', fontweight='bold')
axes[3].legend()

plt.suptitle('Iterative VIF Removal — Optimal Step: Adjusted R²', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('iterative_vif_history.png', dpi=150, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Step 8. Baseline vs Enhanced — Final Training and Evaluation
# Baseline : SELECTED_FEATURES without estimate_msrp
# Enhanced : SELECTED_FEATURES including estimate_msrp
# Adjusted R² is the key comparison metric because the two models differ
# by exactly one feature; a plain R² comparison would always favour Enhanced.
# -----------------------------------------------------------------------------
X_tr_final = X_train[SELECTED_FEATURES].copy()
X_te_final = X_test[SELECTED_FEATURES].copy()
y_tr_final = y_train.copy()
y_te_final = y_test.copy()

y_tr_log = np.log1p(y_tr_final)
y_te_log = np.log1p(y_te_final)

X_tr_base, X_te_base, _, _ = encode_and_scale(
    X_tr_final.drop(columns=['estimate_msrp'], errors='ignore'),
    X_te_final.drop(columns=['estimate_msrp'], errors='ignore'),
    y_tr_log
)
X_tr_enh, X_te_enh, _, _ = encode_and_scale(
    X_tr_final.copy(), X_te_final.copy(), y_tr_log,
    scale_cols=['odometer', 'vehicle_age', "estimate_msrp"]
)


def train_and_evaluate(config_name, X_tr, X_te, y_tr_log, y_te_log):
    """Train Linear Regression and return evaluation metrics + predictions."""
    model = LinearRegression()
    model.fit(X_tr, y_tr_log)
    y_pred = np.clip(np.expm1(model.predict(X_te)), 0, None)
    y_true = np.expm1(y_te_log)

    n  = len(y_true)
    p  = X_te.shape[1]
    r2 = r2_score(y_true, y_pred)

    return {
        'Config'  : config_name,
        'MAE'     : mean_absolute_error(y_true, y_pred),
        'RMSE'    : np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²'      : r2,
        'Adj. R²' : adjusted_r2(r2, n, p),
        'n'       : n,
        'p'       : p,
        'y_true'  : y_true,
        'y_pred'  : y_pred,
        'model'   : model,
        'X_test'  : X_te,
    }


res_base = train_and_evaluate('Baseline', X_tr_base, X_te_base, y_tr_log, y_te_log)
res_enh  = train_and_evaluate('Enhanced', X_tr_enh,  X_te_enh,  y_tr_log, y_te_log)

print(f'\n{"":12} {"MAE":>12} {"RMSE":>12} {"R²":>8} {"Adj. R²":>10} {"Features":>8}')
print('-' * 66)
for res in [res_base, res_enh]:
    print(f'{res["Config"]:12} ${res["MAE"]:>10,.0f} ${res["RMSE"]:>10,.0f} '
          f'{res["R²"]:>8.4f} {res["Adj. R²"]:>10.4f} {res["p"]:>8}')


# -----------------------------------------------------------------------------
# Step 9. Visualization
# -----------------------------------------------------------------------------

# --- Bar chart: 4 metrics side-by-side ---
configs    = ['Baseline', 'Enhanced']
bar_colors = ['#3498DB', '#E74C3C']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metrics = [
    ([res_base['R²'],      res_enh['R²']],      'R² Score',    '{:.4f}',  True),
    ([res_base['Adj. R²'], res_enh['Adj. R²']], 'Adjusted R²', '{:.4f}',  True),
    ([res_base['RMSE'],    res_enh['RMSE']],    'RMSE ($)',    '${:,.0f}', False),
    ([res_base['MAE'],     res_enh['MAE']],     'MAE ($)',     '${:,.0f}', False),
]

for ax, (vals, title, fmt, is_r2) in zip(axes, metrics):
    bars = ax.bar(configs, vals, color=bar_colors, edgecolor='white', width=0.5)
    ax.set_title(title, fontweight='bold', fontsize=12)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                fmt.format(val),
                ha='center', fontweight='bold', fontsize=11)
    if is_r2:
        ax.set_ylim(0, 1)

plt.suptitle('Linear Regression — Baseline vs Enhanced', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('baseline_vs_enhanced.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'R²       : {res_base["R²"]:.4f} → {res_enh["R²"]:.4f}  ({res_enh["R²"]-res_base["R²"]:+.4f})')
print(f'Adj. R²  : {res_base["Adj. R²"]:.4f} → {res_enh["Adj. R²"]:.4f}  ({res_enh["Adj. R²"]-res_base["Adj. R²"]:+.4f})')
print(f'RMSE     : ${res_base["RMSE"]:,.0f} → ${res_enh["RMSE"]:,.0f}  ({res_enh["RMSE"]-res_base["RMSE"]:+,.0f}$)')
print(f'MAE      : ${res_base["MAE"]:,.0f} → ${res_enh["MAE"]:,.0f}  ({res_enh["MAE"]-res_base["MAE"]:+,.0f}$)')

# --- Actual vs Predicted scatter (Enhanced) ---
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(res_enh['y_true'], res_enh['y_pred'], alpha=0.3, s=10, color='#3498DB')
lim = max(res_enh['y_true'].max(), res_enh['y_pred'].max())
ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect Prediction Line')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title(
    f'Actual vs Predicted — Enhanced\n'
    f'R²={res_enh["R²"]:.4f}  Adj.R²={res_enh["Adj. R²"]:.4f}  RMSE=${res_enh["RMSE"]:,.0f}',
    fontweight='bold'
)
ax.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Step 10. Conclusion Summary
# -----------------------------------------------------------------------------
r2_diff     = res_enh['R²']      - res_base['R²']
adj_r2_diff = res_enh['Adj. R²'] - res_base['Adj. R²']
rmse_diff   = res_enh['RMSE']   - res_base['RMSE']
mae_diff    = res_enh['MAE']    - res_base['MAE']
removed_groups = [h['remove_label'] for h in history if h['removed']]

print('=' * 62)
print('           Linear Regression Analysis Conclusion')
print('=' * 62)

print(f'\n[1] Iterative VIF Removal')
print(f'    Initial features : {len(numeric_feats)}')
print(f'    Steps explored   : {len(history)}')
print(f'    Removed groups   : {removed_groups}')
print(f'    Optimal step     : Step {int(best_step["step"])} → {len(SELECTED_NUMERIC)} features + model')
print(f'    Optimal max VIF  : {best_step["max_vif"]:.2f}  (multicollinearity present)')
print(f'    → Coefficient interpretation is unreliable, but prediction metrics remain valid.')

print(f'\n[2] Baseline vs Enhanced')
print(f'    {"":10} {"R²":>8} {"Adj. R²":>10} {"RMSE":>10} {"MAE":>10} {"Features":>8}')
print(f'    {"-" * 58}')
print(f'    {"Baseline":10} {res_base["R²"]:>8.4f} {res_base["Adj. R²"]:>10.4f} '
      f'${res_base["RMSE"]:>9,.0f} ${res_base["MAE"]:>9,.0f} {res_base["p"]:>8}')
print(f'    {"Enhanced":10} {res_enh["R²"]:>8.4f} {res_enh["Adj. R²"]:>10.4f} '
      f'${res_enh["RMSE"]:>9,.0f} ${res_enh["MAE"]:>9,.0f} {res_enh["p"]:>8}')
print(f'    {"Change":10} {r2_diff:>+8.4f} {adj_r2_diff:>+10.4f} '
      f'${rmse_diff:>+9,.0f} ${mae_diff:>+9,.0f}')

print(f'\n[3] Impact of estimate_msrp')
print(f'    Adj. R² change : {adj_r2_diff:+.4f}')
print(f'    RMSE change    : {rmse_diff:+,.0f}$')
print(f'    MAE change     : {mae_diff:+,.0f}$')
if adj_r2_diff > 0.005:
    print('    → Adj. R² significantly improved → estimate_msrp is genuinely useful.')
elif adj_r2_diff > 0:
    print('    → Adj. R² slightly improved → estimate_msrp contributes modest gain.')
else:
    print('    → Adj. R² did not improve → estimate_msrp adds minimal value.')

print('=' * 62)
