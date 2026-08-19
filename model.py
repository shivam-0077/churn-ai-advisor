import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_and_train_model():
    # For demonstration, we use a sample public structure or Telco dataset format
    # You can replace this with a local CSV path: pd.read_csv('data/customer_churn.csv')
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_train.csv" # Placeholder or use local dataset
    
    # Let's create a dummy DataFrame simulating customer metrics if no external file is loaded yet:
    data = {
        'CustomerID': [f'CUST-{i:03d}' for i in range(1, 101)],
        'Tenure': [int(x) for x in pd.Series([12, 2, 24, 6, 36, 1, 48, 8, 15, 3] * 10)],
        'MonthlyCharges': [float(x) for x in pd.Series([45.5, 89.2, 25.0, 75.4, 102.1, 90.0, 30.5, 65.0, 80.0, 95.5] * 10)],
        'SupportTickets': [int(x) for x in pd.Series([0, 3, 1, 4, 0, 5, 1, 2, 3, 4] * 10)],
        'UsageFrequency': [int(x) for x in pd.Series([25, 5, 30, 10, 28, 2, 29, 15, 12, 4] * 10)],
        'Churn': [int(x) for x in pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 1, 1] * 10)]
    }
    df = pd.DataFrame(data)

    # Features and Target
    X = df[['Tenure', 'MonthlyCharges', 'SupportTickets', 'UsageFrequency']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities for the entire dataset
    df['ChurnProbability'] = model.predict_proba(X)[:, 1] * 100
    
    return df, model