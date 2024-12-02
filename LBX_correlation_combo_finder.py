import pandas as pd
from sqlalchemy import create_engine, text
from operator import itemgetter
import itertools
import statsmodels.api as sm
from statsmodels.tools import add_constant
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

engine = create_engine(
    "mssql://@lbx-sqlintel/LBX_Market_Intelligence?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

query = "select * from fcast_base_data_pivoted  where year > 1995 order by Year desc, Month  desc"
df = pd.read_sql_query(sql=text(query), con=engine.connect())

query = "select distinct ItemName from fcast_base_data_unpivoted_tmp where year > 1995 and ItemVarType IN ('D','B') order by ItemName "
dependent_vars = pd.read_sql_query(sql=text(query), con=engine.connect())["ItemName"]

query = "select distinct ItemName from fcast_base_data_unpivoted_tmp where year > 1995 and ItemVarType in ('I','B') order by ItemName "

independent_vars = pd.read_sql_query(sql=text(query), con=engine.connect())["ItemName"]

# Define the range of rows to shift the independent variables
# shift_range = list(range(-24, 0))
shift_range = [-1, -2, -3, -6, -12]
saved_results = []
saved_parameters = []
writeflag = 0
modelid = 0
maxresults = 0.0
# prep the database table that will store the results - R squared and parameters

scale = StandardScaler()

with engine.connect() as connection:
    connection.execute(
        text("DROP TABLE IF EXISTS fcast_correlation_combo_finder_results")
    )
    connection.execute(
        text("DROP TABLE IF EXISTS fcast_correlation_combo_finder_results_params")
    )
    connection.commit()

for combos in range(1, 5, 1):
    for depvar in dependent_vars:
        print(depvar, combos)
        for indepvar in itertools.combinations(independent_vars, combos):

            # define the columns we need in our data
            columns = ("MonthsSince1900",) + (depvar,) + indepvar
            depvar_trend = depvar + "_trend"
            depvar_detrend = depvar + "_detrend"
            indepvar_trend = [s + "_trend" for s in indepvar]
            indepvar_detrend = [s + "_detrend" for s in indepvar]
            indepvar_detrend_scaled = [s + "_detrend_scaled" for s in indepvar]

            # Get the combintations of shifted columns
            shift_combinations = itertools.product(shift_range, repeat=len(indepvar))
            shift_combinations = list(shift_combinations)

            # print(f"Dep Var: {depvar}, Indep Var: {indepvar}")
            for shifts in shift_combinations:
                modelid = modelid + 1

                # copy the main dataframe to get just the columns we need
                shifted_df = df[list(columns)].copy()

                # shift all of the independent variable columns
                for var, shift in zip(indepvar, shifts):
                    # shifted_df[var].shift(shift)
                    shifted_df[var] = shifted_df[var].shift(shift)

                # Drop rows with NaN values due to shifting
                shifted_df = shifted_df.dropna()

                # Get min and max dates
                min_months_since_1900 = shifted_df[["MonthsSince1900"]].min()
                max_months_since_1900 = shifted_df[["MonthsSince1900"]].max()
                num_samples = shifted_df.shape[0]

                # first we have to remove any trend from the dependent variable
                y = shifted_df[[depvar]]
                X = shifted_df[["MonthsSince1900"]]

                model = LinearRegression().fit(X, y)
                shifted_df[depvar_trend] = model.predict(X)
                shifted_df[depvar_detrend] = (
                    shifted_df[depvar] - shifted_df[depvar_trend]
                )

                # Next, do it for all of the independent variables
                for y_i, y_t, y_d, y_s in zip(
                    indepvar, indepvar_trend, indepvar_detrend, indepvar_detrend_scaled
                ):
                    y = shifted_df[[y_i]]
                    X = shifted_df[["MonthsSince1900"]]

                    model = LinearRegression().fit(X, y)
                    shifted_df[y_t] = model.predict(X)

                    shifted_df[y_d] = shifted_df[y_i] - shifted_df[y_t]
                    # scale the data
                    shifted_df[y_s] = scale.fit_transform(shifted_df[[y_d]])

                # print("New iteration:")
                X = add_constant(shifted_df[list(indepvar_detrend_scaled)])
                y = shifted_df[depvar]

                # Fit the regression model
                model = sm.OLS(y, X)
                results = model.fit()

                if abs(results.rsquared) >= 0.3:
                    writeflag = writeflag + 1
                    if results.rsquared > maxresults:
                        print(
                            f"ModelID: {modelid}, Dep Var: {depvar_detrend}, Indep Var: {indepvar_detrend_scaled}, R-squared: {results.rsquared}"
                        )
                        maxresults = results.rsquared
                    # Store the results
                    saved_results.append(
                        {
                            "ModelID": modelid,
                            "Model": "OLS",
                            "MinMonthsSince1900": min_months_since_1900,
                            "MaxMonthsSince1900": max_months_since_1900,
                            "Number of Samples": num_samples,
                            "Dependent Variable": depvar_detrend,
                            "Independent Variable Count": combos,
                            "Independent Variables": str(indepvar_detrend_scaled),
                            "Shifts": str(shifts),
                            "r_squared": results.rsquared,
                        }
                    )
                    # put the coefficients into a separate table, keyed to the results table
                    for xcol, p in zip(X.columns, results.params):
                        saved_parameters.append(
                            {"ModelID": modelid, "IndepVar": xcol, "Coefficient": p}
                        )

            if writeflag > 100:
                print("Writing\n")
                # Store the results for each combination of independent variables independent
                ndf = pd.DataFrame(saved_results)
                paramdf = pd.DataFrame(saved_parameters)
                with engine.connect() as connection:
                    ndf.to_sql(
                        "fcast_correlation_combo_finder_results",
                        connection,
                        if_exists="append",
                        index=False,
                    )
                    connection.commit()
                with engine.connect() as connection:
                    paramdf.to_sql(
                        "fcast_correlation_combo_finder_results_params",
                        connection,
                        if_exists="append",
                        index=False,
                    )
                    connection.commit()

                saved_results = []
                saved_parameters = []

                writeflag = 0
print("Complete!")
