import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics

df = pd.read_csv("car data.csv")

df.drop("Car_Name", axis=1, inplace=True)
df.insert(0, "Age", df["Year"].max() + 1 - df["Year"])
df.drop("Year", axis=1, inplace=True)

sns.set_style("darkgrid")
sns.set_palette(sns.color_palette(["#0055ff", "#ff7000", "#23bf00"]))

ordered_cols = np.concatenate([
    df.select_dtypes(exclude="object").columns.values,
    df.select_dtypes(include="object").columns.values
])

fig, ax = plt.subplots(2, 4, figsize=(15, 7), dpi=100)
for i, col in enumerate(ordered_cols):
    r, c = divmod(i, 4)
    if i < 5:
        sns.boxplot(data=df, y=col, ax=ax[r, c])
    else:
        sns.boxplot(data=df, x=col, y="Selling_Price", ax=ax[r, c])
plt.tight_layout()
plt.show()

outliers = set()
target = "Selling_Price"

for col in df.select_dtypes(include="object").columns:
    for cat in df[col].unique():
        t = df[df[col] == cat]
        q1, q3 = t[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers.update(t[(t[target] < lo) | (t[target] > hi)].index)

for col in df.select_dtypes(exclude="object").columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers.update(df[(df[col] < lo) | (df[col] > hi)].index)

remove_idx = set(df[df["Selling_Price"] > 33].index)
remove_idx.update(df[df["Kms_Driven"] > 400000].index)

df = df.drop(index=list(remove_idx), errors="ignore").reset_index(drop=True)

num_features = ["Present_Price", "Kms_Driven", "Age", "Owner"]
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for i, f in enumerate(num_features):
    r, c = divmod(i, 2)
    ax[r, c].scatter(df[f], df[target])
    ax[r, c].set_xlabel(f)
    ax[r, c].set_ylabel(target)
plt.tight_layout()
plt.show()

cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for i, f in enumerate(cat_cols):
    sns.stripplot(data=df, x=f, y=target, ax=axes[i])
plt.tight_layout()
plt.show()

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.tight_layout()
plt.show()

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE :", metrics.mean_absolute_error(y_test, y_pred))
print("MSE :", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2  :", metrics.r2_score(y_test, y_pred))

pipeline = make_pipeline(StandardScaler(), LinearRegression())
kf = KFold(n_splits=6, shuffle=True, random_state=0)

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=kf,
    scoring=["neg_mean_absolute_error",
             "neg_mean_squared_error",
             "neg_root_mean_squared_error",
             "r2"]
)

print(pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Mean": [
        -scores["test_neg_mean_absolute_error"].mean(),
        -scores["test_neg_mean_squared_error"].mean(),
        -scores["test_neg_root_mean_squared_error"].mean(),
        scores["test_r2"].mean()
    ]
}))

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.plot(kind="bar", figsize=(18,6))
plt.tight_layout()
plt.show()
