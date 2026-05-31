# =============================================================================
# Vehicle Price Adequacy Classification Model Analysis
# Goal  : Classify each vehicle as Underpriced / Fairly Priced / Overpriced
#         using residuals from the Enhanced Linear Regression model
# Flow  : Load → Residual labels → Feature align → CV comparison →
#         Best model → Validate → Apply to 54,805-row test set → Tendency
# =============================================================================

import platform
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

if platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42
N_FOLDS      = 5


# -----------------------------------------------------------------------------
# Step 1. Load Data
# df_train  : 7,192 rows — residual calculation + classification training
# df_test   : 54,805 rows (already scaled) — final classification target
# -----------------------------------------------------------------------------
df_train = pd.read_csv('preprocessed_vehicles.csv')
print('Training data shape:', df_train.shape)

df_test = pd.read_csv('preprocessed_vehicle_classification_scaled.csv')
print('Test data shape:', df_test.shape)


# -----------------------------------------------------------------------------
# Step 2. Residual Calculation and Label Generation (Enhanced Regression)
# Enhanced model (with estimate_msrp) gives the most accurate predictions,
# so its residuals serve as the basis for price adequacy labels.
# Label boundaries use the 33rd/67th percentiles of residuals so that
# each class gets ~33% of samples, avoiding class imbalance from the start.
# -----------------------------------------------------------------------------
target_col  = 'price'
exclude_reg = ['price', 'model']
reg_feature_cols = [c for c in df_train.columns if c not in exclude_reg]

X_reg = df_train[reg_feature_cols].copy()
y_reg = df_train[target_col].copy()

numeric_cols = ['condition', 'odometer', 'vehicle_age', 'estimate_msrp']
scaler_reg   = StandardScaler()
X_reg[numeric_cols] = scaler_reg.fit_transform(X_reg[numeric_cols])

lr_model = LinearRegression()
lr_model.fit(X_reg, y_reg)
y_pred_reg = lr_model.predict(X_reg)

residuals = y_reg - y_pred_reg
print(f'Residual Std Dev = {residuals.std():,.0f}')
print(f'R² = {lr_model.score(X_reg, y_reg):.4f}')

p33 = np.percentile(residuals, 33)
p67 = np.percentile(residuals, 67)
print(f'\nBoundaries:  33rd pct = ${p33:,.0f}  |  67th pct = ${p67:,.0f}')

# 0 = Underpriced, 1 = Fairly Priced, 2 = Overpriced
price_class = pd.Series(1, index=df_train.index, name='price_class')
price_class[residuals < p33] = 0
price_class[residuals > p67] = 2

df_train['residual']    = residuals.values
df_train['price_class'] = price_class.values

print('\nLabel distribution:')
label_names = {0: 'Underpriced', 1: 'Fairly Priced', 2: 'Overpriced'}
for cls, cnt in df_train['price_class'].value_counts().sort_index().items():
    print(f'  {cls} ({label_names[cls]}): {cnt}  ({cnt/len(df_train)*100:.1f}%)')


# -----------------------------------------------------------------------------
# Step 3. Check Class Distribution
# Percentile-based labeling guarantees ~33% per class; confirming visually
# before proceeding (no SMOTE needed if balanced).
# -----------------------------------------------------------------------------
counts       = df_train['price_class'].value_counts().sort_index()
colors       = ['#3498DB', '#2ECC71', '#E74C3C']
class_labels = ['Underpriced\n(0)', 'Fairly Priced\n(1)', 'Overpriced\n(2)']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bars = axes[0].bar(class_labels, counts.values, color=colors, edgecolor='white', linewidth=0.8)
for bar, cnt in zip(bars, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
        f'{cnt}\n({cnt/len(df_train)*100:.1f}%)',
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_ylim(0, counts.max() * 1.2)
axes[0].grid(axis='y', alpha=0.4)

axes[1].pie(
    counts.values, labels=class_labels, colors=colors,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
)
axes[1].set_title('Class Distribution (Pie Chart)', fontsize=13, fontweight='bold')

plt.suptitle('Price Adequacy Class Distribution (N=7,192)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
print('Class distribution is balanced — no SMOTE needed.')


# -----------------------------------------------------------------------------
# Step 4. Feature Alignment and Preprocessing
# Classification features exclude estimate_msrp so the model can also run
# on the 54,805-row test set which has no MSRP estimates.
# Columns present in one dataset but not the other are reconciled here.
# -----------------------------------------------------------------------------
test_feature_cols = [c for c in df_test.columns if c not in ['model_encoded', 'price_log']]

exclude_clf = ['price', 'model', 'estimate_msrp', 'residual', 'price_class',
               'manufacturer_chrysler']
train_feature_cols = [c for c in df_train.columns if c not in exclude_clf]

# Add any test-only columns to training data (filled with 0)
for col in test_feature_cols:
    if col not in train_feature_cols:
        df_train[col] = 0
        train_feature_cols.append(col)

train_feature_cols       = sorted(train_feature_cols)
test_feature_cols_sorted = sorted(test_feature_cols)

assert train_feature_cols == test_feature_cols_sorted, 'Feature mismatch between train and test sets'
print(f'Final feature count: {len(train_feature_cols)}')

X_clf      = df_train[train_feature_cols].copy()
y_clf      = df_train['price_class'].copy()
X_test_clf = df_test[test_feature_cols_sorted].copy()

# Scale numeric columns — fit on training data only
clf_numeric_cols = ['condition', 'odometer', 'vehicle_age']
scaler_clf = StandardScaler()
X_clf[clf_numeric_cols] = scaler_clf.fit_transform(X_clf[clf_numeric_cols])

print(f'X_clf shape     : {X_clf.shape}')
print(f'X_test_clf shape: {X_test_clf.shape}')


# -----------------------------------------------------------------------------
# Step 5. Train / Validation Split (80:20, stratified)
# stratify=y preserves class ratios in both splits.
# -----------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)
print(f'Train: {X_train.shape[0]}  |  Validation: {X_val.shape[0]}')


# -----------------------------------------------------------------------------
# Step 6. Define Classification Models
# Four algorithms covering linear, tree-based, distance-based, and ensemble.
# -----------------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'KNN'                : KNeighborsClassifier(n_neighbors=7),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, max_depth=15,
                                                   random_state=RANDOM_STATE, n_jobs=-1),
}
print('Models:', list(models.keys()))


# -----------------------------------------------------------------------------
# Step 7. Stratified K-Fold Cross Validation
# StratifiedKFold keeps the ~33/33/33 class ratio in every fold, giving
# stable estimates. F1-macro is the primary metric because all three classes
# are equally important — a plain accuracy score would hide per-class failure.
# -----------------------------------------------------------------------------
skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = []

for name, model in models.items():
    acc_scores = cross_val_score(model, X_clf, y_clf, cv=skf, scoring='accuracy', n_jobs=-1)
    f1_scores  = cross_val_score(model, X_clf, y_clf, cv=skf, scoring='f1_macro',  n_jobs=-1)
    cv_results.append({
        'Model'   : name,
        'Acc Mean': acc_scores.mean(),
        'Acc Std' : acc_scores.std(),
        'F1 Mean' : f1_scores.mean(),
        'F1 Std'  : f1_scores.std(),
    })
    print(f'[{name}]  Accuracy={acc_scores.mean():.4f}±{acc_scores.std():.4f}  '
          f'F1-macro={f1_scores.mean():.4f}±{f1_scores.std():.4f}')

cv_df = pd.DataFrame(cv_results).sort_values('F1 Mean', ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Step 8. Model Performance Comparison Visualization
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
model_names = cv_df['Model'].tolist()
colors_bar  = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71']

for ax, metric, ylabel, title in zip(
    axes,
    ['Acc Mean', 'F1 Mean'],
    ['Accuracy', 'F1-score (macro)'],
    ['K-Fold CV Accuracy (5-Fold)', 'K-Fold CV F1-score Macro (5-Fold)']
):
    std_key = metric.replace('Mean', 'Std')
    bars = ax.bar(model_names, cv_df[metric], yerr=cv_df[std_key],
                  color=colors_bar[:len(model_names)], capsize=5,
                  edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, cv_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.4)

plt.suptitle('Classification Model K-Fold CV Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 9. Optimal Model Selection and Validation
# Pick the highest F1-macro model, retrain on the 80% split, evaluate on 20%.
# -----------------------------------------------------------------------------
best_model_name = cv_df.iloc[0]['Model']
best_model      = models[best_model_name]
print(f'Best model: {best_model_name}')

best_model.fit(X_train, y_train)
y_val_pred = best_model.predict(X_val)

val_acc = accuracy_score(y_val, y_val_pred)
val_f1  = f1_score(y_val, y_val_pred, average='macro')
print(f'\nValidation (N={len(y_val)}):  Accuracy={val_acc:.4f}  F1-macro={val_f1:.4f}')

# Per-class report only (strip accuracy / macro avg / weighted avg lines)
report = classification_report(
    y_val, y_val_pred,
    target_names=['Underpriced', 'Fairly Priced', 'Overpriced']
)
filtered_lines = [
    line for line in report.split('\n')
    if not any(x in line for x in ['accuracy', 'macro avg', 'weighted avg'])
]
print('\nClassification Report (Validation Set):')
print('\n'.join(filtered_lines))


# -----------------------------------------------------------------------------
# Step 10. Confusion Matrix
# Absolute counts (left) and row-normalised proportions (right).
# Fairly Priced typically shows the lowest recall since it sits between
# the two boundary classes and shares residual space with both.
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_val, y_val_pred)
class_labels_str = ['Underpriced', 'Fairly\nPriced', 'Overpriced']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels_str, yticklabels=class_labels_str,
            linewidths=0.5, ax=axes[0])
axes[0].set_title(f'Confusion Matrix — {best_model_name}\n(Absolute Counts)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Class', fontsize=11)
axes[0].set_ylabel('Actual Class', fontsize=11)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_labels_str, yticklabels=class_labels_str,
            linewidths=0.5, ax=axes[1])
axes[1].set_title(f'Confusion Matrix — {best_model_name}\n(Row Proportions)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Class', fontsize=11)
axes[1].set_ylabel('Actual Class', fontsize=11)

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 11. Apply Classification to 54,805-Row Test Data
# Retrain on all 7,192 rows first to maximise information before inference.
# -----------------------------------------------------------------------------
best_model.fit(X_clf, y_clf)
y_test_pred = best_model.predict(X_test_clf)

df_test['price_class'] = y_test_pred
df_test['price_label'] = df_test['price_class'].map(
    {0: 'Underpriced', 1: 'Fairly Priced', 2: 'Overpriced'}
)

test_counts = df_test['price_class'].value_counts().sort_index()
print('Test data classification results (N=54,805):')
for cls, cnt in test_counts.items():
    lbl = {0: 'Underpriced', 1: 'Fairly Priced', 2: 'Overpriced'}[cls]
    print(f'  {cls} ({lbl}): {cnt:,}  ({cnt/len(df_test)*100:.1f}%)')


# -----------------------------------------------------------------------------
# Step 12. Visualize Test Data Classification Results
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#3498DB', '#2ECC71', '#E74C3C']
class_labels_test = ['Underpriced\n(0)', 'Fairly Priced\n(1)', 'Overpriced\n(2)']

bars = axes[0].bar(class_labels_test, test_counts.values, color=colors,
                   edgecolor='white', linewidth=0.8)
for bar, cnt in zip(bars, test_counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
        f'{cnt:,}\n({cnt/len(df_test)*100:.1f}%)',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )
axes[0].set_title('Predicted Class Distribution (54,805 rows)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Vehicle Count', fontsize=11)
axes[0].set_ylim(0, test_counts.max() * 1.2)
axes[0].grid(axis='y', alpha=0.4)

axes[1].pie(
    test_counts.values, labels=class_labels_test, colors=colors,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
)
axes[1].set_title('Predicted Class Ratio (54,805 rows)', fontsize=13, fontweight='bold')

plt.suptitle(f'[{best_model_name}] Price Adequacy — Test Data Results',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 13. Price Range Class Distribution Analysis
# Split price_log into 5 quantile bins to see whether under/over-pricing
# concentrates in specific price tiers.
# -----------------------------------------------------------------------------
df_test['price_bin'] = pd.qcut(
    df_test['price_log'], q=5,
    labels=['Very Low', 'Low', 'Mid', 'High', 'Very High']
)

price_class_dist = (
    df_test.groupby(['price_bin', 'price_class'], observed=True)
    .size().unstack(fill_value=0)
    .apply(lambda x: x / x.sum(), axis=1)
)
price_class_dist.columns = ['Underpriced', 'Fairly Priced', 'Overpriced']

fig, ax = plt.subplots(figsize=(10, 6))
price_class_dist.plot(
    kind='bar', stacked=True,
    color=['#3498DB', '#2ECC71', '#E74C3C'],
    edgecolor='white', linewidth=0.5, ax=ax
)
ax.set_title('Class Distribution by Price Range (54,805 rows)', fontsize=13, fontweight='bold')
ax.set_xlabel('Price Range (log price quantile)', fontsize=11)
ax.set_ylabel('Class Ratio', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Class', loc='upper right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 14. Dataset Pricing Tendency Summary
# Identify the dominant class in the 54,805-row test set.
# -----------------------------------------------------------------------------
label_map      = {0: 'Underpriced', 1: 'Fairly Priced', 2: 'Overpriced'}
dominant_class = test_counts.idxmax()
dominant_label = label_map[dominant_class]
dominant_pct   = test_counts.max() / len(df_test) * 100

tendency_msg = {
    0: ('Underpriced',    'Vehicles in this dataset are generally priced below fair market value.\n'
                          'This dataset represents a market with many undervalued vehicles.'),
    1: ('Fairly Priced',  'Vehicles in this dataset are generally priced at fair market value.\n'
                          'This dataset represents a well-priced market.'),
    2: ('Overpriced',     'Vehicles in this dataset are generally priced above fair market value.\n'
                          'This dataset represents a market with many overvalued vehicles.'),
}

print('=' * 55)
print('  Class proportions in 54,805-row test data')
print('=' * 55)
for cls, cnt in test_counts.items():
    marker = '  ◀ Most' if cls == dominant_class else ''
    print(f'  {label_map[cls]:15s}: {cnt:,}  ({cnt/len(df_test)*100:.1f}%){marker}')
print('=' * 55)
print(f'\n→ Dominant class: {dominant_label} ({dominant_pct:.1f}%)')
print(f'\n  {tendency_msg[dominant_class][1]}')
