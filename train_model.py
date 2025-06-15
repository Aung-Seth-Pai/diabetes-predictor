from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# load datasets
diabetes = load_diabetes()
# print(diabetes.DESCR)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# train random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
mean_squared_error_value = mean_squared_error(y_test, y_pred)
r2_score_value = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mean_squared_error_value}")
print(f"R^2 Score: {r2_score_value}")

# save model to disk
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, "diabetes_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
# confirm model saved
if os.path.exists(model_path):
    print(f"Model saved to {model_path}")

# # load model from disk
# with open(model_path, "rb") as f:
#     loaded_model = pickle.load(f)
# # confirm model loaded
# if loaded_model:
#     print("Model loaded successfully from disk.")
