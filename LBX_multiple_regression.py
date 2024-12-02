
import pandas as pd
from sqlalchemy import create_engine, text
from operator import itemgetter

engine = create_engine("mssql://@lbx-sqlintel/LBX_Market_Intelligence?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")
query = "select * from fcast_base_data_pivoted  where year > 2009 order by Year desc, Month  desc"

df = pd.read_sql_query(sql=text(query), con=engine.connect())

import itertools
import statsmodels.api as sm
from statsmodels.tools import add_constant

# Load your dataframe
# df = pd.read_csv('your_data.csv')

# Replace 'y' with the name of your dependent variable column
dependent_var = 'RetailUnitsDirt'

# Replace ['x1', 'x2', 'x3'] with the names of your inependent variable columns
independent_vars = ['EFFR','HOUSTNSA','HOUST1FNSA']

fetch_vars = independent_vars.copy()

fetch_vars.append(dependent_var)

df = df[fetch_vars]

# Define the range of rows to shift the independent variables
shift_range = list(range(-4, 1))

print(len(independent_vars))
# Generate all combinations of shifts for the independent variables
shift_combinations = itertools.product(shift_range, repeat=len(independent_vars))
shift_combinations = list(shift_combinations)

saved_results = []

# Iterate through all combinations of shifts and calculate multiple regression and R-squared

#shifted_df = df.copy()

for shifts in shift_combinations :
    #print("start shift")
    shifted_df = df.copy()
    for var, shift in zip(independent_vars, shifts):
        #shifted_df[var].shift(shift)
        shifted_df[var] = shifted_df[var].shift(shift)

    # Drop rows with NaN values due to shifting
    shifted_df = shifted_df.dropna()
    print("New iteration:")
    print(shifts)
    # Prepare the dependent and independent variables

    X = add_constant(shifted_df[independent_vars])
    y = shifted_df[dependent_var]

    #print(shifted_df)
    # Fit the multiple regression model
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.rsquared)

    # Calculate R-squared
    #r_squared = model.rsquared
    
    # Store the results
    saved_results.append({
         'shifts': shifts,
         'r_squared': results.rsquared,
           'res': results,
            'data':shifted_df.copy()
     })
    #print(var, shift)

    
# Sort the results by R-squared in descending order
saved_results.sort(key=lambda x: x['r_squared'], reverse=True)

# Print the best model
best_model = saved_results[0]
print(f"Best model shifts: {best_model['shifts']}")
print(f"Best model R-squared: {best_model['r_squared']}")
print(best_model['res'].summary())
print(best_model['data'])