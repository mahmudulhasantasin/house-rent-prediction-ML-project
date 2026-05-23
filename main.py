import pandas as pd

# Load dataset
df = pd.read_csv("House_Rent_Dataset.csv")

# ----------------------------
# 1. DROP USELESS COLUMNS
# ----------------------------
df = df.drop(["Posted On", "Point of Contact"], axis=1)

# ----------------------------
# 2. CLEAN FLOOR COLUMN
# ----------------------------
def extract_floor(x):
    try:
        return int(str(x).split(" ")[0])
    except:
        return 0

df["Floor"] = df["Floor"].apply(extract_floor)

# ----------------------------
# 3. HANDLE MISSING VALUES
# ----------------------------
df = df.dropna()

# ----------------------------
# 4. REMOVE OUTLIERS (IMPORTANT)
# ----------------------------
df = df[df["Rent"] < 200000]   # remove extreme rents
df = df[df["Size"] < 5000]     # remove huge houses

# ----------------------------
# 5. DEFINE FEATURES & TARGET
# ----------------------------
X = df.drop("Rent", axis=1)
y = df["Rent"]

# ----------------------------
# 6. ENCODE CATEGORICAL DATA
# ----------------------------
X = pd.get_dummies(X)

# ----------------------------
# 7. TRAIN TEST SPLIT
# ----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 8. FEATURE SCALING
# ----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 9. TRAIN MODEL
# ----------------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# 10. EVALUATION
# ----------------------------
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))