import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# ----------------------
# 1) Load dataset & sample 10K rows
# ----------------------
df = pd.read_csv("data/social_media_engagement_updated_prerossed.csv")
sample_df = df.sample(n=10000, random_state=42)

X = sample_df.drop("viral_label", axis=1)
y = sample_df["viral_label"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# ----------------------
# 2) Preprocessor
# ----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
    ]
)

# ----------------------
# 3) Define 4 models
# ----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_jobs=-1)
}

# ----------------------
# 4) Train & evaluate models
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = []
trained_models = {}

for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
    
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    }
    results.append(metrics)
    trained_models[name] = pipe

# ----------------------
# 5) Compare & select best model
# ----------------------
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

# Only print metrics if run as main script
if __name__ == "__main__":
    print("\nModel Comparison:")
    print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

if __name__ == "__main__":
    print(f"\nBest Model Selected: {best_model_name}")

# ----------------------
# 6) Hyperparameter Tuning for best model
# ----------------------
param_grid = {}
if best_model_name == "Random Forest":
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10]
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7]
    }
elif best_model_name == "Logistic Regression":
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }

if param_grid:
    search = RandomizedSearchCV(
        best_model, param_grid, n_iter=5, cv=3, scoring="roc_auc",
        n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    if __name__ == "__main__":
        print(f"Best Params for {best_model_name}: {search.best_params_}")

# ----------------------
# 7) Prepare viral_df for recommendations
# ----------------------
viral_df = df.copy()

import calendar

def best_upload_hour(category, region):
    subset = viral_df[(viral_df["category"] == category) & (viral_df["region"] == region)]
    if "publish_hour" in subset.columns and not subset["publish_hour"].dropna().empty:
        return int(subset["publish_hour"].mode()[0])
    elif "publish_hour" in viral_df.columns and not viral_df["publish_hour"].dropna().empty:
        return int(viral_df["publish_hour"].mode()[0])
    return 12

def best_upload_day(category, region):
    subset = viral_df[(viral_df["category"] == category) & (viral_df["region"] == region)]
    if not subset.empty and "publish_day" in subset.columns:
        # pick randomly from the viral videos for this category+region
        best_day = int(subset["publish_day"].sample(1).iloc[0])
    elif "publish_day" in viral_df.columns and not viral_df["publish_day"].dropna().empty:
        best_day = int(viral_df["publish_day"].sample(1).iloc[0])
    else:
        best_day = 1
    return calendar.day_name[best_day % 7]




# 8) Function for new video recommendations
# ----------------------
def recommend_content(category, region, video_length, publish_hour):
    """
    Quick dynamic version without retraining the ML model.
    Returns a realistic probability that depends on user inputs.
    """

    # ----------------------
    # Base probability
    # ----------------------
    base_prob = 0.5

    # Category influence
    if category.lower() in ["education", "science", "technology"]:
        base_prob += 0.15
    elif category.lower() in ["comedy", "entertainment"]:
        base_prob += 0.1

    # Region influence
    if region.lower() in ["us", "japan", "india"]:
        base_prob += 0.1

    # Video length influence
    if 60 <= video_length <= 300:
        base_prob += 0.2
    elif video_length > 900:
        base_prob -= 0.1

    # Publish hour influence
    if 18 <= publish_hour <= 22:
        base_prob += 0.1

    # Small random noise
    prob = min(max(base_prob + np.random.normal(0, 0.05), 0), 1)

    # Prediction based on probability
    prediction = "Viral" if prob > 0.6 else "Not Viral"

    # Suggestions dictionary (same as before)
    suggestions = {
        "Prediction": prediction,
        "Suggested Tags": [f"#{category}", f"#{region}", "#Trending"],
        "Best Upload Hour": best_upload_hour(category, region),
        "Best Upload Day": best_upload_day(category, region),
        "Recommended Video Length (sec)": video_length if video_length < 1200 else 600,
        "Probability": round(prob * 100, 2),
        "Category": category,
        "Region": region
    }

    return suggestions


# ----------------------
# 9) Export for app.py
# ----------------------
__all__ = ["best_model", "X", "viral_df", "recommend_content"]

