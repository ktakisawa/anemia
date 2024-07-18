import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'path_to_your_local_file.csv'  # Change this to your file path
df = pd.read_csv(file_path)

# Select relevant features
features = ['%Red Pixel', '%Green Pixel', '%Blue Pixel', 'Hb', 'Anaemic']
df = df[features]

# Split the data into two subsets: Anaemic (Yes) and Non-Anaemic (No)
anaemic_yes = df[df['Anaemic'] == 'Yes']
anaemic_no = df[df['Anaemic'] == 'No']

# Normalize the data
scaler = StandardScaler()
anaemic_yes[['%Red Pixel', '%Green Pixel', '%Blue Pixel']] = scaler.fit_transform(anaemic_yes[['%Red Pixel', '%Green Pixel', '%Blue Pixel']])
anaemic_no[['%Red Pixel', '%Green Pixel', '%Blue Pixel']] = scaler.fit_transform(anaemic_no[['%Red Pixel', '%Green Pixel', '%Blue Pixel']])

# Prepare the data for regression
X_yes = anaemic_yes[['%Red Pixel', '%Green Pixel', '%Blue Pixel']]
y_yes = anaemic_yes['Hb']

X_no = anaemic_no[['%Red Pixel', '%Green Pixel', '%Blue Pixel']]
y_no = anaemic_no['Hb']

# Initialize and fit the linear regression model for Anaemic: Yes
model_yes = LinearRegression()
model_yes.fit(X_yes, y_yes)

# Initialize and fit the linear regression model for Anaemic: No
model_no = LinearRegression()
model_no.fit(X_no, y_no)

# Get the coefficients and intercepts
coefficients_yes = model_yes.coef_
coefficients_no = model_no.coef_
intercept_yes = model_yes.intercept_
intercept_no = model_no.intercept_

# Display the results
print("Anaemic (Yes) Model Coefficients:", coefficients_yes)
print("Anaemic (Yes) Model Intercept:", intercept_yes)
print("Non-Anaemic (No) Model Coefficients:", coefficients_no)
print("Non-Anaemic (No) Model Intercept:", intercept_no)
