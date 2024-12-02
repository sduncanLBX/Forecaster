import pandas as pd
from sqlalchemy import create_engine, text
from operator import itemgetter
import itertools
import statsmodels.api as sm
from statsmodels.tools import add_constant
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def init_database():
    engine = create_engine(
        "mssql://@lbx-sqlintel/LBX_Market_Intelligence?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
    )
    return engine


def drop_tables(engine):
    with engine.connect() as connection:
        connection.execute(
            text("DROP TABLE IF EXISTS fcast_correlation_combo_finder_results")
        )
        connection.execute(
            text("DROP TABLE IF EXISTS fcast_correlation_combo_finder_results_params")
        )
        connection.commit()


def write_df(engine, dataframeToWrite, TablenameToWriteTo):
    with engine.connect() as connection:
        dataframeToWrite.to_sql(
            TablenameToWriteTo,
            connection,
            if_exists="append",
            index=False,
        )
        connection.commit()


def write_results():
    global saved_parameters, saved_results, writeflag
    print("-->Writing", end="")
    # Store the results for each combination of independent variables independent
    ndf = pd.DataFrame(saved_results)
    paramdf = pd.DataFrame(saved_parameters)
    write_df(midb, ndf, "fcast_correlation_combo_finder_results")
    write_df(midb, paramdf, "fcast_correlation_combo_finder_results_params")
    saved_results = []
    saved_parameters = []
    writeflag = 0
    print("...done writing", end="")


def store_results():
    # Store the results
    saved_results.append(
        {
            "ModelID": modelid,
            "Model": "OLS",
            "MinMonthsSince1900": min_months_since_1900,
            "MaxMonthsSince1900": max_months_since_1900,
            "Number of Samples": num_samples,
            "Dependent Variable": depvar,
            "Independent Variable Count": combos,
            "Independent Variables": str(indepvar),
            "Shifts": str(shifts),
            "r_squared": rsquared,
        }
    )
    # put the coefficients into a separate table, keyed to the results table
    for xcol, p in zip(X.columns, coeffients):
        saved_parameters.append(
            {"ModelID": modelid, "IndepVar": xcol, "Coefficient": p}
        )


def process_variable_set(DependentVar, IndependentVars):
    global saved_results, saved_parameters, df, maxresults, modelid, ResultsSavedCount, shifts, rsquared, X, min_months_since_1900, max_months_since_1900, num_samples, coeffients, writeflag, depvar_detrend, indepvar_detrend_scaled
    shift_range = [-1, -2, -3, -6, -12]
    shift_range = range(-24, 1, 1)
    # define the columns we need in our data
    columns = ("MonthsSince1900",) + (DependentVar,) + IndependentVars
    depvar_trend = DependentVar + "_trend"
    depvar_detrend = DependentVar + "_detrend"
    indepvar_trend = [s + "_trend" for s in IndependentVars]
    indepvar_detrend = [s + "_detrend" for s in IndependentVars]
    indepvar_detrend_scaled = [s + "_detrend_scaled" for s in IndependentVars]

    # Get the combintations of shifted columns
    shift_combinations = itertools.product(shift_range, repeat=len(IndependentVars))
    shift_combinations = list(shift_combinations)

    # print(f"Dep Var: {depvar}, Indep Var: {indepvar}")
    for shifts in shift_combinations:
        modelid += 1

        # copy the main dataframe to get just the columns we need
        shifted_df = df[list(columns)].copy()

        # # shift all of the independent variable columns
        for var, shift in zip(IndependentVars, shifts):
            # shifted_df[var].shift(shift)
            shifted_df[var] = shifted_df[var].shift(shift)

        # for var in IndependentVars:
        #     # shifted_df[var].shift(shift)
        #     shifted_df[var] = shifted_df[var].shift(shifts)

        # Drop rows with NaN values due to shifting
        shifted_df = shifted_df.dropna()

        # Get min and max dates
        min_months_since_1900 = shifted_df[["MonthsSince1900"]].min()
        max_months_since_1900 = shifted_df[["MonthsSince1900"]].max()
        num_samples = shifted_df.shape[0]

        # # first we have to remove any trend from the dependent variable
        # y = shifted_df[[DependentVar]]
        # X = shifted_df[["MonthsSince1900"]]

        # model = LinearRegression().fit(X, y)
        # shifted_df[depvar_trend] = model.predict(X)
        # shifted_df[depvar_detrend] = shifted_df[DependentVar] - shifted_df[depvar_trend]

        # # Next, do it for all of the independent variables
        # for y_i, y_t, y_d, y_s in zip(
        #     indepvar, indepvar_trend, indepvar_detrend, indepvar_detrend_scaled
        # ):
        #     y = shifted_df[[y_i]]
        #     X = shifted_df[["MonthsSince1900"]]

        #     model = LinearRegression().fit(X, y)
        #     shifted_df[y_t] = model.predict(X)

        #     shifted_df[y_d] = shifted_df[y_i] - shifted_df[y_t]
        #     # scale the data
        #     shifted_df[y_s] = scale.fit_transform(shifted_df[[y_d]])

        # # print("New iteration:")
        # X = add_constant(shifted_df[list(indepvar_detrend_scaled)])
        # y = shifted_df[DependentVar]

        # X = add_constant(shifted_df[[IndependentVars]])
        X = shifted_df[list(IndependentVars)]
        y = shifted_df[DependentVar]

        # Fit the regression model
        # model = sm.OLS(y, X)
        # results = model.fit()
        # rsquared = results.rsquared

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, random_state=0, test_size=0.5
        )
        model = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )
        model.fit(Xtrain, ytrain)
        ymodel = model.predict(Xtest)
        rsquared = r2_score(ytest, ymodel)
        coeffients = model.named_steps['regressor'].coef_

        print(
            "ModelID:{:7d}, Depth:{:1d} Saved: {:6d}, Dep Var: {:>25.25}, Indep Var: {:<75.75}, Max R-squared: {:1.3f}\r".format(
                modelid,
                combos,
                ResultsSavedCount,
                depvar,
                str(indepvar),
                maxresults,
            ),
            end="",
        )

        if rsquared > maxresults:
            maxresults = rsquared

        if rsquared >= 0:
            writeflag += 1
            ResultsSavedCount += 1
            store_results()


midb = init_database()

query = "select * from fcast_base_data_pivoted  where year > 1995 order by Year desc, Month  desc"
df = pd.read_sql_query(sql=text(query), con=midb.connect())

query = "select distinct ItemName from fcast_base_data_unpivoted_tmp where year > 1995 and ItemVarType IN ('D','B') order by ItemName "
dependent_vars = pd.read_sql_query(sql=text(query), con=midb.connect())["ItemName"]

query = "select distinct ItemName from fcast_base_data_unpivoted_tmp where year > 1995 and ItemVarType in ('I','B') order by ItemName "

independent_vars = pd.read_sql_query(sql=text(query), con=midb.connect())["ItemName"]

# Define the range of rows to shift the independent variables

saved_results = []
saved_parameters = []
writeflag = 0
modelid = 0
maxresults = 0.0
ResultsSavedCount = 0
# prep the database table that will store the results - R squared and parameters

scale = StandardScaler()

drop_tables(midb)

for combos in range(1, 2, 1):
    for depvar in dependent_vars:
        for indepvar in itertools.combinations(independent_vars, combos):
            process_variable_set(depvar, indepvar)
            #if writeflag > 10:
            #    write_results()
if writeflag > 0:
    write_results()
print("Analysis Complete!")
