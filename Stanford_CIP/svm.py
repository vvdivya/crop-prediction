import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Optional: print column names to verify structure
print("Columns in dataset:", df.columns.tolist())

# Feature and target split
x = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
model = SVC()
model.fit(train_x, train_y)

# Accuracy check
predicted = model.predict(test_x)
acc = accuracy_score(test_y, predicted)
print("Accuracy:", acc)

# Save model
pickle.dump(model, open("svm_model.sav", 'wb'))

# Optional: Example of a safe prediction input
sample_input = {
    'N': 90,
    'P': 40,
    'K': 40,
    'temperature': 20,
    'humidity': 80,
    'ph': 7.0,
    'rainfall': 200
}

# Convert to DataFrame with column names matching training data
input_df = pd.DataFrame([sample_input])
predicted_crop = model.predict(input_df)
print("Predicted crop for sample input:", predicted_crop[0])


from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# Assuming you have a trained SVM model called 'model'
svc = SVC(kernel='linear')  # Using a linear kernel for demonstration

# RFE
rfe = RFE(estimator=svc, n_features_to_select=1)
rfe.fit(train_x, train_y)

# Print feature ranking
feature_ranking = rfe.ranking_
#feature_names = ['soil_ph', 'phosphorus', 'potassium']
feature_names = ['N','P','K', 'temperature','humidity', 'ph', 'rainfall']
for rank, name in zip(feature_ranking, feature_names):
    print(f"Feature {name} has importance rank: {rank}")

# Optionally, plot the feature importance (simplified for small datasets)
import matplotlib.pyplot as plt
plt.barh(feature_names, feature_ranking)
plt.xlabel("Feature importance rank (1 is most important)")
plt.ylabel("Features")
plt.title("Feature Importance Ranking")
plt.show()