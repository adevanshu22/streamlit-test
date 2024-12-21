import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
# from sklearn.metrics import mean_squared_error, r2_score
# import plotly.express as px

# Synthetic generation of data examples for training the model
# def generate_house_data(n_samples=100):
#     from sklearn.datasets import make_hastie_10_2
#     np.random.seed(42)
#     size = np.random.normal(1500, 500, n_samples)
#     price = size * 100 + np.random.normal(0, 10000, n_samples)
#     return pd.DataFrame({'size_sqft': size, 'price': price})

# Function for instantiating and training linear regression model
def train_model():
    # df = generate_data()

    X, y = make_hastie_10_2(1000, random_state=0)
    
    # Train-test data splitting
    # X = df[['size_sqft']]
    # y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.2, random_state=42)
    
    # Train the model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0)
    model.fit(X_train, y_train)
    
    return model

# Streamlit User Interface for Deployed Model
def main():
    st.title('🧑‍💻 A sample Classifer')
    st.write('Classify as +1 or -1 basis numerical inputs')
    
    # Train model
    model = train_model()
    
    # User input

    X = np.zeros(2)

    X[0] = st.slider("feature 1", -3, 3, value = 0)
    X[1] = st.slider("feature 2", -3, 3, value = 0)
    
    if st.button('Predict Outcome'):
        # Perform prediction

        probability = model.predict_proba(X.reshape(1, -1))
        prediction = model.predict(X.reshape(1, -1))
        
        # Show result
        st.success(f'Estimated Probablity: {round(probability[0][1] * 100, 2)}%')

        st.success(f'Estimated Outcome: {prediction[0]:}')

if __name__ == '__main__':
    main()