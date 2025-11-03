# Next Steps After Feature Extraction

## Complete ML Pipeline Workflow

### ✅ STEP 1: Feature Extraction (DONE)
- Extracted 54 features from stress-strain curves
- Created DataFrame with features and alloy composition

---

### 📋 STEP 2: Data Preprocessing (NEXT)

#### 2.1 Handle Null/Missing Values
**Status: You mentioned wanting to do this**

```python
# Fill nulls with median for each feature
for feature in feature_cols:
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)
```

**Why median?**
- Robust to outliers (better than mean)
- Preserves feature distribution
- Works well with skewed data

**Alternative options:**
- Mean (if data is normally distributed)
- Mode (for categorical features)
- Forward fill / Backward fill (for time series)
- Remove rows (if <5% missing)

---

#### 2.2 Feature Scaling/Normalization
**IMPORTANT: Most ML algorithms need this!**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Features only, not targets
```

**Why scale?**
- Features have different units and ranges
- UTS might be 1000+ MPa, while ratios are 0-1
- Ensures all features contribute equally

**Options:**
- **StandardScaler** (mean=0, std=1) - Best for most cases
- **MinMaxScaler** (0-1 range) - Good for neural networks
- **RobustScaler** (median, IQR) - Better for outliers

---

#### 2.3 Handle Outliers (Optional but Recommended)
```python
from scipy import stats

# Option 1: Remove outliers using Z-score
z_scores = np.abs(stats.zscore(X))
X_clean = X[(z_scores < 3).all(axis=1)]

# Option 2: Cap outliers at percentiles
for col in X.columns:
    q99 = X[col].quantile(0.99)
    q01 = X[col].quantile(0.01)
    X[col] = X[col].clip(lower=q01, upper=q99)
```

---

### 📊 STEP 3: Feature Analysis

#### 3.1 Check Feature Distributions
```python
import matplotlib.pyplot as plt

# Check if features are normally distributed
df[feature_cols].hist(bins=30, figsize=(20, 15))
plt.tight_layout()
plt.show()
```

#### 3.2 Feature Correlation Analysis
```python
# Check for highly correlated features (potential redundancy)
correlation_matrix = df[feature_cols].corr()

# Find highly correlated pairs (|r| > 0.95)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.95:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr
            ))
```

#### 3.3 Feature Importance (After initial model)
```python
# Train a quick model to see feature importance
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_scaled, y)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

---

### 🔀 STEP 4: Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split features and targets
X = df[feature_cols]  # 54 features
y = df[alloy_cols]    # Alloy composition (Al, Cr, Fe, etc.)

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=None  # Can't stratify for regression
)
```

**Important considerations:**
- **For 392 samples**: 80/20 split = ~314 train, ~78 test
- Use `random_state` for reproducibility
- Consider stratified split if classification

---

### 🤖 STEP 5: Model Selection & Training

#### Option A: Multi-Output Regression (Predict all elements at once)
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Model options:
# 1. Random Forest
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
)

# 2. Gradient Boosting
model = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        l2_regularization=0.1,
        max_iter=200,
        random_state=42
    )
)

# Train
model.fit(X_train, y_train)
```

#### Option B: Separate Models for Each Element
```python
models = {}
for element in alloy_cols:
    models[element] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    models[element].fit(X_train, y_train[element])
```

#### Option C: Neural Network (if you have more data later)
```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
```

---

### 📈 STEP 6: Model Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics for each element
for i, element in enumerate(alloy_cols):
    print(f"\n{element}:")
    print(f"  Train R²: {r2_score(y_train.iloc[:, i], y_train_pred[:, i]):.4f}")
    print(f"  Test R²:  {r2_score(y_test.iloc[:, i], y_test_pred[:, i]):.4f}")
    print(f"  Test MAE: {mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i]):.4f}%")
    print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])):.4f}%")
```

**Key Metrics:**
- **R² Score**: How much variance is explained (1.0 = perfect)
- **MAE**: Mean absolute error in percentage
- **RMSE**: Root mean squared error (penalizes large errors more)

---

### 🎯 STEP 7: Hyperparameter Tuning (Optional but Recommended)

```python
from sklearn.model_selection import GridSearchCV

# Example: Tune Random Forest
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [5, 10, 15],
    'estimator__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

### 📊 STEP 8: Feature Selection (If Needed)

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel

# Option 1: Select top K features
selector = SelectKBest(f_regression, k=40)
X_selected = selector.fit_transform(X_train, y_train)

# Option 2: Use model-based selection
selector = SelectFromModel(
    RandomForestRegressor(n_estimators=100),
    threshold='median'
)
X_selected = selector.fit_transform(X_train, y_train)
```

---

## Recommended Order of Steps:

1. ✅ **Handle null values** (fill with median) ← YOU ARE HERE
2. ✅ **Feature scaling** (StandardScaler)
3. ✅ **Train/test split** (80/20)
4. ✅ **Train baseline model** (Random Forest or Gradient Boosting)
5. ✅ **Evaluate model** (R², MAE, RMSE)
6. ✅ **Feature importance analysis** (identify most important features)
7. ✅ **Optional: Feature selection** (remove low-importance features)
8. ✅ **Optional: Hyperparameter tuning** (improve performance)
9. ✅ **Final model evaluation** on test set

---

## Quick Start Code Template:

```python
# After handling nulls...
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Prepare features and targets
X = df[feature_cols]
y = df[alloy_cols]

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Train model
model = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        l2_regularization=0.1,
        max_iter=200,
        random_state=42
    )
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
```

---

## Priority: Do These First

1. **Handle null values** ← Critical
2. **Feature scaling** ← Critical
3. **Train/test split** ← Critical
4. **Train baseline model** ← See if it works

Then iterate on improvements!

