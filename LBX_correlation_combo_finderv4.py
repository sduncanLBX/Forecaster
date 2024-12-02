import pandas as pd
from sqlalchemy import create_engine, text
import itertools
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
writeflag = 0  # if we save results we increment this value so we know we have something to write
modelid = 0  # Used as the primary key in the fcast_results table, and to join to the other tables
maxresults = 0.0  # the highest r-squared value
ResultsSavedCount = 0  # the number of results that have been saved
DependentVar = ""
IndependentVars = ""
mseval = 0.0  # the mean squared error score for the current model
rsquared = 0.0  # the r-squared value fro the current model

# set variables used to define what analysis takes place
PipelineContents = [("scaler", StandardScaler()), ("regressor", LinearRegression())]

# These variables define how the system runs
# starting data year defines the minimum year of data that is selected for all variables.
starting_data_year = "1995"

# Set to True to save parameters (coefficients) in a separate table
SaveParameters = True

# Set to True to save predicted values in a separate table - both actuals and FCast values
SavePredictions = True

# number of independent variables to combine in multi-variate regression. Comma-separated list
CombinationsToCalculate = [2]

# Set to True to drop and re-create the datatables
ManageTables = 'Clean'

# Shift ranges define whether independent variables will be shifted before analysis.
# A shift of -12 means January, 2023 independent variables will be correlated to January 2024 dependent variable data
shift_range = [-3, -6, -12,-24]


# We won't save or write model data for models with RSquared below this number
RSquaredTarget = 0.20
MSETarget = 99999999999999999.0

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


def get_data(engine):
    global dependent_vars, independent_vars, df,dependent_vars_12mo_avg
    query = (
        "select * from fcast_base_data_pivoted  where year >= "
        + starting_data_year
        + " order by Year desc, Month  desc"
    )
    df = pd.read_sql_query(sql=text(query), con=engine.connect())

    query = (
        "select distinct ItemName from fcast_base_data_unpivoted_tmp where year >= "
        + starting_data_year
        + " and ItemVarType IN ('D','B') and ItemName like '%12mo_avg%' order by ItemName "
    )
    dependent_vars_12mo_avg = pd.read_sql_query(sql=text(query), con=engine.connect())["ItemName"]

    query = (
            "select distinct ItemName as Variable from fcast_base_data_unpivoted_tmp where year >= "
            + starting_data_year
            + " and ItemVarType in ('D','B') order by ItemName "
        )
    dependent_vars = pd.read_sql_query(sql=text(query), con=engine.connect())["Variable"]



    query = (
            "select distinct ItemName as Variable from fcast_base_data_unpivoted_tmp where year >= "
            + starting_data_year
            + " and ItemVarType in ('I','B') order by ItemName "
        )

    independent_vars = pd.read_sql_query(sql=text(query), con=engine.connect())[
        "Variable"
    ]


def manage_tables(engine):
    if ManageTables == 'Drop':
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS fcast_results"))
            connection.execute(text("DROP TABLE IF EXISTS fcast_results_params"))

            connection.execute(text("DROP TABLE IF EXISTS fcast_results_predictions"))
            connection.commit()

    if ManageTables == 'Clean':
        with engine.connect() as connection:
            connection.execute(text("truncate table fcast_results_params"))
            connection.execute(text("truncate table fcast_results_predictions"))
            connection.execute(text("delete from fcast_results where [Independent Variable Count] > 1"))
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
    columns = ("MonthsSince1900",) + (DependentVar,) + IndependentVars

    # Get the combintations of shifted columns
    shift_combinations = itertools.product(shift_range, repeat=len(IndependentVars))
    shift_combinations = list(shift_combinations)

    for shifts in shift_combinations:
        modelid += 1

        # copy the main dataframe to get just the columns we need
        ShiftedData = df[list(columns)].copy()

        # # shift all of the independent variable columns
        for var, shift in zip(IndependentVars, shifts):
            ShiftedData[var] = ShiftedData[var].shift(shift)

        # Make a copy of the shifted data to train/test with, with NaN rows removed
        ShiftedTrainData = ShiftedData.copy()
        ShiftedTrainData.dropna(inplace=True)

        # Make a copy of the shifted data to make a full prediction from, with NaN rows removed only using the independent variable columns
        ShiftedPredictionData = ShiftedData.copy()
        ShiftedPredictionData.dropna(inplace=True, subset=IndependentVars)

        # Get min and max dates
        min_months_since_1900 = ShiftedTrainData["MonthsSince1900"].min()
        max_months_since_1900 = ShiftedTrainData["MonthsSince1900"].max()
        num_samples = ShiftedTrainData.shape[0]

        # Set X, y data
        X = ShiftedTrainData[list(IndependentVars)]
        y = ShiftedTrainData[DependentVar]

        # split into training and testing sets
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

        # add the model id as a column, so we have a key when we write this data to the DB
        ShiftedPredictionData["ModelID"] = modelid

        # Rename the dependent variable column to 'Actual' for easy graphing
        ShiftedPredictionData.rename(inplace=True, columns={DependentVar: "Actual"})

        # Get rid of any future-year rows that are empty for whatever reason
        CleanedPredictionResult = ShiftedPredictionData[["Actual", "Fcast"]].dropna()

        # Calc the mean squared error for the entire prediction set
        mseval = mean_squared_error(
            CleanedPredictionResult[["Actual"]], CleanedPredictionResult[["Fcast"]]
        )

        # Print a status message to the terminal while we're running to we know what's going on.
        print(
            "ModelID:{:7d}, Depth:{:1d} Saved: {:6d}, Dep Var: {:>25.25}, Indep Var: {:<75.75}, Max R-squared: {:1.3f}\r".format(
                modelid,
                combos,
                ResultsSavedCount,
                DependentVar,
                str(IndependentVars),
                maxresults,
            ),
            end="",
        )

        # If we have a new high r-squared value, set it.
        if rsquared > maxresults:
            maxresults = rsquared

        # If the model scores high enough, save it and set the write flag
        if (rsquared >= RSquaredTarget) or (mseval < MSETarget):
            writeflag += 1
            ResultsSavedCount += 1
            store_results()

def get_target_rsquared(var, engine):
    # query = (
    #         "select top 1 Max(r_squared) as MaxRSquared from fcast_results where [Dependent Variable] like '%"+var+"%' and [Independent Variable Count] = 1"

    #     )
    # return  pd.read_sql_query(sql=text(query), con=engine.connect())["MaxRSquared"]
    with engine.connect() as my_connection:
        return my_connection.execute(text("select top 1 Max(r_squared) as MaxRSquared from fcast_results where [Dependent Variable] like '%"+var+"%' and [Independent Variable Count] = 1")).scalar()

def get_target_mse(var, engine):
    # query = (
    #         "select top 1 Max(mse) as MinMSE from fcast_results where [Dependent Variable] like '%"+var+"%' and [Independent Variable Count] = 1"

    #     )
    # return  pd.read_sql_query(sql=text(query), con=engine.connect())["MinMSE"]

    with engine.connect() as my_connection:
        return my_connection.execute(text("select top 1 Max(mse) as MinMSE from fcast_results where [Dependent Variable] like '%"+var+"%' and [Independent Variable Count] = 1")).scalar()

def get_invependent_vars_from_dependent_var(var, engine):
    query = (
            "select distinct Variable from fcast_high_scoring_vars where [Dependent Variable] = '"+ var +"'"
        )

    return pd.read_sql_query(sql=text(query), con=engine.connect())[
        "Variable"
    ]



# Set up the DB connection
midb = init_database()

# Get all the data out of the DB
get_data(midb)

# drop the tables (there's a switch inside the function to see if we actually drop)
manage_tables(midb)

combos = 1
shift_range = range(-24, 0, 1)
for DependentVar in dependent_vars_12mo_avg:
    for IndependentVars in independent_vars:
        process_variable_set()
        if writeflag > RecordsToSaveBeforeWrite:
            write_results()
if writeflag > 0:
     write_results()
print("Single Variable Analysis Complete! Yay")

# Now we begin multi-variable!

for combos in CombinationsToCalculate:
    for DependentVar in dependent_vars_12mo_avg:
        MSETarget = get_target_mse(DependentVar,midb)
        RSquaredTarget = get_target_rsquared(DependentVar, midb)
        NewIndependentVars = get_invependent_vars_from_dependent_var(DependentVar, midb)
        for IndependentVars in itertools.combinations(NewIndependentVars, combos):
            process_variable_set()
            if writeflag > RecordsToSaveBeforeWrite:
                write_results()
if writeflag > 0:
    write_results()
print("Analysis Complete! Yay")
