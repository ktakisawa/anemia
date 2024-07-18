from sklearn.linear_model import LinearRegression
import numpy as np

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

# Get the coefficients
coefficients_yes = model_yes.coef_
coefficients_no = model_no.coef_

# Get the intercepts
intercept_yes = model_yes.intercept_
intercept_no = model_no.intercept_

coefficients_yes, intercept_yes, coefficients_no, intercept_no
