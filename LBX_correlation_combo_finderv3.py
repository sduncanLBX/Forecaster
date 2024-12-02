import pandas as pd
from sqlalchemy import create_engine, text
from operator import itemgetter
import itertools
import statsmodels.api as sm
from statsmodels.tools import add_constant
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# initialize some global variables
saved_results = []
saved_parameters = []
saved_predictions = pd.DataFrame
writeflag = 0 # if we save results we increment this value so we know we have something to write
modelid = 0 # Used as the primary key in the fcast_results table, and to join to the other tables 
maxresults = 0.0 # the highest r-squared value
ResultsSavedCount = 0 # the number of results that have been saved
DependentVar = ""
IndependentVars = ""
mseval = 0.0 # the mean squared error score for the current model
rsquared = 0.0 # the r-squared value fro the current model

# set variables used to define what analysis takes place
PipelineContents = [("scaler", StandardScaler()), ("regressor", LinearRegression())]

# These variables define how the system runs
# starting data year defines the minimum year of data that is selected for all variables.
starting_data_year = "1995"

# Set to True to save parameters (coefficients) in a separate table
SaveParameters = False

# Set to True to save predicted values in a separate table - both actuals and FCast values
SavePredictions = False

# number of independent variables to combine in multi-variate regression. Comma-separated list
CombinationsToCalculate = [1]

# Set to True to drop and re-create the datatables
DropTables = True

# Shift ranges define whether independent variables will be shifted before analysis.
# A shift of -12 means January, 2023 independent variables will be correlated to January 2024 dependent variable data
# shift_range = [0,-1,-2,-3, -6, -12,-24]
shift_range = range(-24, 0, 1)

# We won't save or write model data for models with RSquared below this number
RSquaredSaveThreshold = 0.3

# We will accumulate this many records before writing to the database, to minimize the impact of network latency
RecordsToSaveBeforeWrite = 5000

# Where do we pull our independent variables from? If set to 'HighScoring' it will pull them from the
# SQL view fcast_high_scoring_vars which uses the single variables stats to pull the top 20 best peforming
# variables for each dependent variable, both by r squared and mse scores. If blank or anything else it
# pulls all the variables in the data.
PullIndpendentVariablesFrom = "HighScoring"

#####################################################################################################################
#
#  End of Variables
#
#####################################################################################################################


# if we're saving the predicted dataset, we have to write with each save because the records don't accumulate from one model to another
if SavePredictions:
    RecordsToSaveBeforeWrite = 0


def init_database():
    engine = create_engine(
        "mssql://@lbx-sqlintel/LBX_Market_Intelligence?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
    )
    return engine


def drop_tables(engine):
    if DropTables:
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS fcast_results"))
            connection.execute(text("DROP TABLE IF EXISTS fcast_results_params"))

            connection.execute(text("DROP TABLE IF EXISTS fcast_results_predictions"))
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
    global saved_parameters, saved_results, writeflag, saved_predictions
    print("-->Writing", end="")
    # Store the results for each combination of independent variables independent
    ndf = pd.DataFrame(saved_results)
    write_df(midb, ndf, "fcast_results")
    saved_results = []

    if SaveParameters:
        paramdf = pd.DataFrame(saved_parameters)
        write_df(midb, paramdf, "fcast_results_params")
        saved_parameters = []

    if SavePredictions:
        # predictiondf = pd.DataFrame(saved_predictions)
        write_df(midb, saved_predictions, "fcast_results_predictions")
        saved_predictions = pd.DataFrame

    writeflag = 0
    print("...done writing", end="")


def store_results():
    global saved_predictions, SaveParameters, SavePredictions
    # Store the results
    saved_results.append(
        {
            "ModelID": modelid,
            "Model": "OLS",
            "MinMonthsSince1900": min_months_since_1900,
            "MaxMonthsSince1900": max_months_since_1900,
            "Data Starting Year": starting_data_year,
            "Number of Samples": num_samples,
            "Dependent Variable": DependentVar,
            "Independent Variable Count": combos,
            "Independent Variables": str(IndependentVars),
            "Pipeline:": str(PipelineContents),
            "Shifts": str(shifts),
            "r_squared": rsquared,
            "mse": mseval,
        }
    )

    # put the coefficients into a separate table, keyed to the results table
    if SaveParameters:
        for xcol, p in zip(X.columns, coeffients):
            saved_parameters.append(
                {"ModelID": modelid, "IndepVar": xcol, "Coefficient": p}
            )
    if SavePredictions:
        saved_predictions = ShiftedPredictionData[
            ["ModelID", "MonthsSince1900", "Actual", "Fcast"]
        ]


def process_variable_set():
    global saved_results, saved_parameters, df, mseval, maxresults, modelid, ResultsSavedCount, shifts, rsquared, X, min_months_since_1900, max_months_since_1900, num_samples, coeffients, writeflag, depvar_detrend, indepvar_detrend_scaled, ShiftedPredictionData

    # define the columns we need in our data
    # print(IndependentVars)
    columns = ("MonthsSince1900",) + (DependentVar,) + IndependentVars

    # Get the combintations of shifted columns
    shift_combinations = itertools.product(shift_range, repeat=len(IndependentVars))
    shift_combinations = list(shift_combinations)

    # print(f"Dep Var: {depvar}, Indep Var: {indepvar}")
    for shifts in shift_combinations:
        modelid += 1

        # copy the main dataframe to get just the columns we need
        ShiftedData = df[list(columns)].copy()

        # # shift all of the independent variable columns
        for var, shift in zip(IndependentVars, shifts):
            # shifted_df[var].shift(shift)
            ShiftedData[var] = ShiftedData[var].shift(shift)

        # for var in IndependentVars:
        #     # shifted_df[var].shift(shift)
        #     shifted_df[var] = shifted_df[var].shift(shifts)

        # Drop rows with NaN values due to shifting
        # shifted_df = shifted_df.dropna()

        ShiftedTrainData = ShiftedData.copy()
        ShiftedTrainData.dropna(inplace=True)
        ShiftedPredictionData = ShiftedData.copy()
        ShiftedPredictionData.dropna(inplace=True, subset=IndependentVars)

        # Get min and max dates
        min_months_since_1900 = ShiftedTrainData["MonthsSince1900"].min()
        max_months_since_1900 = ShiftedTrainData["MonthsSince1900"].max()
        num_samples = ShiftedTrainData.shape[0]

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
        X = ShiftedTrainData[list(IndependentVars)]
        y = ShiftedTrainData[DependentVar]

        # Fit the regression model
        # model = sm.OLS(y, X)
        # results = model.fit()
        # rsquared = results.rsquared

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, random_state=0, test_size=0.5
        )

        model = Pipeline(PipelineContents)
        model.fit(Xtrain, ytrain)
        ymodel = model.predict(Xtest)
        rsquared = r2_score(ytest, ymodel)

        coeffients = model.named_steps["regressor"].coef_
        Xpredict = ShiftedPredictionData[list(IndependentVars)]
        ShiftedPredictionData["Fcast"] = model.predict(Xpredict)
        ShiftedPredictionData["ModelID"] = modelid
        ShiftedPredictionData.rename(inplace=True, columns={DependentVar: "Actual"})
        CleanedPredictionResult = ShiftedPredictionData[["Actual", "Fcast"]].dropna()
        mseval = mean_squared_error(
            CleanedPredictionResult[["Actual"]], CleanedPredictionResult[["Fcast"]]
        )

        print(
            "ModelID:{:7d}, Depth:{:1d} Saved: {:6d}, Dep Var: {:>25.25}, Indep Var: {:<75.75}, Max R-squared: {:1.3f}\r".format(
                modelid,
                combos,
                ResultsSavedCount,
                depvar,
                str(IndependentVars),
                maxresults,
            ),
            end="",
        )

        if rsquared > maxresults:
            maxresults = rsquared

        if rsquared >= RSquaredSaveThreshold:
            writeflag += 1
            ResultsSavedCount += 1
            store_results()


midb = init_database()

query = (
    "select * from fcast_base_data_pivoted  where year >= "
    + starting_data_year
    + " order by Year desc, Month  desc"
)
df = pd.read_sql_query(sql=text(query), con=midb.connect())

query = (
    "select distinct ItemName from fcast_base_data_unpivoted_tmp where year >= "
    + starting_data_year
    + " and ItemVarType IN ('D','B') order by ItemName "
)
dependent_vars = pd.read_sql_query(sql=text(query), con=midb.connect())["ItemName"]

if PullIndpendentVariablesFrom == "HighScoring":
    query = "select distinct Variable from fcast_high_scoring_vars"
else:
    query = (
        "select distinct ItemName as Variable from fcast_base_data_unpivoted_tmp where year >= "
        + starting_data_year
        + " and ItemVarType in ('I','B') order by ItemName "
    )

independent_vars = pd.read_sql_query(sql=text(query), con=midb.connect())["Variable"]

drop_tables(midb)
for combos in CombinationsToCalculate:
    for depvar in dependent_vars:
        DependentVar = depvar
        for IndependentVars in itertools.combinations(independent_vars, combos):
            process_variable_set()
            if writeflag > RecordsToSaveBeforeWrite:
                write_results()
if writeflag > 0:
    write_results()
print("Analysis Complete!")
