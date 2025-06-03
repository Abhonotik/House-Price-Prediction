import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load training data (make sure 'train.csv' is in the same folder)
df = pd.read_csv('train.csv')

# Choose a few simple features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = df[features]
y = df['SalePrice']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as house_price_model.pkl")
