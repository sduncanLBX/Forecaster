# v7 Adding the ability to divide the independent variables into categories, and picking 1 variable from each category as a combination
# v8 Added separate table for independent variables

import pandas as pd
import datetime
from sqlalchemy import create_engine, text
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# initialize some global variables
table_prefix = "fcast_v8_results"
results_tablename = table_prefix + "_results"
params_tablename = table_prefix + "_params"
predictions_tablename = table_prefix + "_predictions"
ivars_tablename = table_prefix + "_independent_vars"
saved_results = []
saved_parameters = []
saved_predictions = pd.DataFrame
saved_independent_vars = []
writeflag = 0  # if we save results we increment this value so we know we have something to write
modelid = 0  # Used as the primary key in the fcast_results table, and to join to the other tables
maxresults = 0.0  # the highest r-squared value
ResultsSavedCount = 0  # the number of results that have been saved
DependentVar = ""
IndependentVars = ""
mseval = 0.0  # the mean squared error score for the current model
rsquared = 0.0  # the r-squared value fro the current model
test_rsquared = 0  # higher is better
test_mape = 0  # Lower is better
test_mse = 0  # lower is better
test_mae = 0  # Lower is better
predict_rsquared = 0  # higher is better
predict_mape = 0  # Lower is better
predict_mse = 0  # lower is better
predict_mae = 0  # Lower is better
target_mape = 2
target_rsquared = 0


# set variables used to define what analysis takes place
PipelineContents = [
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression(fit_intercept=True)),
]

# These variables define how the system runs
# starting data year defines the minimum year of data that is selected for all variables.
starting_data_year = "2014"

# Set to True to save parameters (coefficients) in a separate table
SaveParameters = False

# Set to True to save predicted values in a separate table - both actuals and FCast values
SavePredictions = True

# Set to True to drop and re-create the datatables
ManageTables = ""

# Shift ranges define whether independent variables will be shifted before analysis.
# A shift of -12 means January, 2023 independent variables will be correlated to January 2024 dependent variable data
shift_range = [3, 6, 12]
# shift_range = range(1,25, 1)

# Process notes - inserted in each model resutl record
process_note = "This time with no dropped months"

#####################################################################################################################
#
#  End of Variables
#
#####################################################################################################################


def get_date_from_monthssince1900(MonthsSince1900):
    if MonthsSince1900 % 12 ==0:
        month = 12
    else:
        month = MonthsSince1900 % 12
    year = int(((MonthsSince1900 - month)/12)+1900)

    return datetime.date(year, month, 15)

def init_database():
    engine = create_engine(
        "mssql://@lbx-sqlintel/LBX_Market_Intelligence?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
    )
    return engine


def get_data(engine):
    global dependent_vars, independent_vars, df, dependent_vars_12mo_avg, independent_vars_12mo

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
    dependent_vars_12mo_avg = pd.read_sql_query(sql=text(query), con=engine.connect())[
        "ItemName"
    ]

    query = (
        "select distinct ItemName as Variable from fcast_base_data_unpivoted_tmp where year >= "
        + starting_data_year
        + " and ItemVarType in ('D','B') order by ItemName "
    )
    dependent_vars = pd.read_sql_query(sql=text(query), con=engine.connect())[
        "Variable"
    ]

    query = (
        "select distinct ItemName as Variable from fcast_base_data_unpivoted_tmp where year >= "
        + starting_data_year
        + " and ItemVarType in ('I','B') order by ItemName "
    )

    independent_vars = pd.read_sql_query(sql=text(query), con=engine.connect())[
        "Variable"
    ]

    query = (
        "select distinct ItemName as Variable from fcast_base_data_unpivoted_tmp where year >= "
        + starting_data_year
        + " and ItemVarType in ('I','B') and ItemName like '%12mo_avg%' order by ItemName "
    )

    independent_vars_12mo = pd.read_sql_query(sql=text(query), con=engine.connect())[
        "Variable"
    ]


def manage_tables(engine):

    if ManageTables == "Drop":
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS " + results_tablename))
            connection.execute(text("DROP TABLE IF EXISTS " + params_tablename))

            connection.execute(text("DROP TABLE IF EXISTS " + predictions_tablename))
            connection.commit()

    if ManageTables == "Clean":
        with engine.connect() as connection:
            connection.execute(text("truncate table " + params_tablename))
            connection.execute(text("truncate table " + predictions_tablename))
            connection.execute(
                text(
                    "delete from "
                    + results_tablename
                    + " where [Independent Variable Count] > 1"
                )
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
    global saved_parameters, saved_results, writeflag, saved_predictions, results_tablename, params_tablename, predictions_tablename
    print("-->Writing", end="")
    # Store the results for each combination of independent variables independent
    ndf = pd.DataFrame(saved_results)
    write_df(midb, ndf, results_tablename)
    saved_results = []

    ivdf = pd.DataFram(saved_independent_vars)
    write_df(midb, ivdf, ivars_tablename)
    saved_independent_vars = []

    if SaveParameters:
        paramdf = pd.DataFrame(saved_parameters)
        write_df(midb, paramdf, params_tablename)
        saved_parameters = []

    if SavePredictions:
        # predictiondf = pd.DataFrame(saved_predictions)
        write_df(midb, saved_predictions, predictions_tablename)
        saved_predictions = pd.DataFrame

    writeflag = 0
    print("...done writing", end="")


def write_dependentvar_finished(var, engine):
    finished_depvars = pd.DataFrame
    finished_depvars["DependentVar"].append(var)
    with engine.connect() as connection:
        finished_depvars.to_sql(
            table_prefix + "_finished_dep_vars",
            connection,
            if_exists="append",
            index=False,
        )
        connection.commit()


def store_results():
    global saved_results, saved_predictions, SaveParameters, SavePredictions
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
            "Test RSquare": test_rsquared,
            "Test MAPE": test_mape,
            "Test MAE": test_mae,
            "Test MSE": test_mse,
            "Predict RSquare": predict_rsquared,
            "Predict MAPE": predict_mape,
            "Predict MAE": predict_mae,
            "Predict MSE": predict_mse,
            "r_squared": rsquared,
            "mse": mseval,
            "Note": process_note,
            "ScriptFilename": __file__,
            "ForescastLeadTimeMonths": min(
                shifts
            ),  # because they are negative values, max is actually min.
        }
    )

    for var, shift in zip(IndependentVars, shifts):
        global saved_independent_vars
        saved_independent_vars.append(
            {"ModelID": modelid, "IndependentVar": var, "MonthsShifted": shift}
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


def save_result_set(result_set):
    # Store the results

    print("-->Writing", end="")
    saved_results.append(
        {
            "ModelID": result_set["modelid"],
            "Model": "OLS",
            "Training Start Date": result_set["Training Start Date"],
            "Training End Date": result_set["Training End Date"],
            "Months dropped from training":result_set['drop_months'],
            "Data Starting Year": starting_data_year,
            "Number of Samples": result_set["num_samples"],
            "Dependent Variable": result_set["dependent_variable"],
            "Independent Variable Count": len(result_set["independent_variable_list"]),
            "Independent Variables": str(result_set["independent_variable_list"]),
            "Pipeline:": str(PipelineContents),
            "Shifts": str(result_set["shifts"]),
            "Test RSquare": result_set["test_rsquared"],
            "Test MAPE": result_set["test_mape"],
            "Test MAE": result_set["test_mae"],
            "Test MSE": result_set["test_mse"],
            "Predict RSquare": result_set["predict_rsquared"],
            "Predict MAPE": result_set["predict_mape"],
            "Predict MAE": result_set["predict_mae"],
            "Predict MSE": result_set["predict_mse"],
            "Note": process_note,
            "ScriptFilename": __file__,
            "ForescastLeadTimeMonths": min(
                result_set["shifts"]
            ),  # because they are negative values, max is actually min.
        }
    )
    ndf = pd.DataFrame(saved_results)
    write_df(midb, ndf, results_tablename)

    for var, shift in zip(result_set["independent_variable_list"], result_set["shifts"]):
        global saved_independent_vars
        saved_independent_vars.append(
            {
                "ModelID": result_set["modelid"],
                "IndependentVar": var,
                "MonthsShifted": shift,
            }
        )
    ivdf = pd.DataFrame(saved_independent_vars)
    write_df(midb, ivdf, ivars_tablename)

    saved_predictions = result_set["prediction_set"][
            ["ModelID", "Date", "Actual", "Fcast"]
        ]
    # predictiondf = pd.DataFrame(saved_predictions)
    write_df(midb, saved_predictions, predictions_tablename)
    saved_predictions = pd.DataFrame

    print("...done writing", end="")


def process_variable_set():
    global df, mseval, maxresults, modelid, target_mape, ResultsSavedCount, target_mape, target_rsquared, shifts, rsquared, X, min_months_since_1900, max_months_since_1900, num_samples, coeffients, writeflag, depvar_detrend, indepvar_detrend_scaled, ShiftedPredictionData, test_rsquared, test_mse, test_mae, test_mape, predict_mae, predict_mae, predict_mse, predict_mape, predict_rsquared

    # define the columns we need in our data
    columns = ("MonthsSince1900",) + (DependentVar,) + IndependentVars
    # print(columns)
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
        test_rsquared = r2_score(ytest, ymodel)
        test_mape = mean_absolute_percentage_error(ytest, ymodel)
        test_mse = mean_squared_error(ytest, ymodel)
        test_mae = mean_absolute_error(ytest, ymodel)
        # coeffients = model.named_steps["regressor"].coef_
        if test_rsquared > target_rsquared:
            target_rsquared = test_rsquared

        if test_mape <= target_mape:  # or (mseval < MSETarget)
            target_mape = test_mape
            if SavePredictions:
                # Make a copy of the shifted data to make a full prediction from, with NaN rows removed only using the independent variable columns
                ShiftedPredictionData = ShiftedData.copy()
                ShiftedPredictionData.dropna(inplace=True, subset=IndependentVars)
                Xpredict = ShiftedPredictionData[list(IndependentVars)]
                ShiftedPredictionData["Fcast"] = model.predict(Xpredict)

                # add the model id as a column, so we have a key when we write this data to the DB
                ShiftedPredictionData["ModelID"] = modelid

                # Rename the dependent variable column to 'Actual' for easy graphing
                ShiftedPredictionData.rename(
                    inplace=True, columns={DependentVar: "Actual"}
                )

                # Get rid of any future-year rows that are empty for whatever reason
                CleanedPredictionResult = ShiftedPredictionData[
                    ["Actual", "Fcast"]
                ].dropna()

                # Calc the mean squared error for the entire prediction set
                predict_mse = mean_squared_error(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                predict_mape = mean_absolute_percentage_error(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                predict_mae = mean_absolute_error(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                predict_rsquared = r2_score(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                store_results()
                write_results()
            writeflag += 1
            ResultsSavedCount += 1
            store_results()

        # Print a status message to the terminal while we're running to we know what's going on.
        print(
            "ModelID:{:7d}, Depth:{:1d} Saved: {:6d}, Dep Var: {:>25.25}, Indep Var: {:<100.100}, MAPE: {:1.3f}, R^2: {:1.3f}\r".format(
                modelid,
                combos,
                ResultsSavedCount,
                DependentVar,
                str(IndependentVars),
                target_mape,
                target_rsquared,
            ),
            end="",
        )

        # If we have a new high r-squared value, set it.
        # if rsquared > maxresults:
        #     maxresults = rsquared


def process_var_list(independent_variable_list, dependent_variable, shift_range):
    global df, modelid

    model_results = {}
    # define the columns we need in our data
    columns = ["Date", dependent_variable] + independent_variable_list
    # print(columns)
    # Get the combintations of shifted columns
    shift_combinations = itertools.product(
        shift_range, repeat=len(independent_variable_list)
    )
    shift_combinations = list(shift_combinations)
    mape_limit = 2

    for shifts in shift_combinations:
        modelid += 1

        # copy the main dataframe to get just the columns we need
        ShiftedData = df[list(columns)].copy()

        # # shift all of the independent variable columns
        for var, shift in zip(independent_variable_list, shifts):
            ShiftedData[var] = ShiftedData[var].shift(
                -shift
            )  # make shift negative because of how the data is sorted.

        # Make a copy of the shifted data to train/test with, with NaN rows removed
        ShiftedTrainData = ShiftedData.copy()
        ShiftedTrainData.dropna(inplace=True)

        # Make sure the data is sorted from latest to earliest to ensure shifts and removed rows work as expected
        ShiftedTrainData.sort_values('Date', ascending=False)

        # Limit the data available to train by removing desired number of rows from top of dataframe
        # it's the top because we are sorting
        drop_months = 0 
        ShiftedTrainData = ShiftedTrainData.iloc[drop_months:]



        # Set X, y data
        X = ShiftedTrainData[list(independent_variable_list)]
        y = ShiftedTrainData[dependent_variable]

        # split into training and testing sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, random_state=0, test_size=0.5
        )

        model = Pipeline(PipelineContents)
        model.fit(Xtrain, ytrain)
        ymodel = model.predict(Xtest)

        test_mape = mean_absolute_percentage_error(ytest, ymodel)

        if test_mape < mape_limit:
            mape_limit = test_mape
            # store all the metrics
            model_results["test_rsquared"] = r2_score(ytest, ymodel)
            model_results["test_mape"] = test_mape
            model_results["test_mse"] = mean_squared_error(ytest, ymodel)
            model_results["test_mae"] = mean_absolute_error(ytest, ymodel)
            model_results["coeffients"] = model.named_steps["regressor"].coef_
            model_results["shifts"] = shifts

            # Get min and max dates
            model_results["Training Start Date"] = ShiftedTrainData[
                "Date"
            ].min()
            model_results["Training End Date"] = ShiftedTrainData[
                "Date"
            ].max()
            model_results["num_samples"] = ShiftedTrainData.shape[0]
     

            # Print a status message to the terminal while we're running to we know what's going on.
            print(
                "ModelID:{:7d}, Dep Var: {:>20.20}, Indep Var: {:<100.100}, MAPE: {:1.3f}, R^2: {:1.3f}\r".format(
                    modelid,
                    dependent_variable,
                    str(independent_variable_list),
                    mape_limit,
                    target_rsquared,
                ),
                end="",
            )

            if SavePredictions:
                # Make a copy of the shifted data to make a full prediction from, with NaN rows removed only using the independent variable columns
                ShiftedPredictionData = ShiftedData.copy()
                ShiftedPredictionData.dropna(
                    inplace=True, subset=independent_variable_list
                )
                Xpredict = ShiftedPredictionData[list(independent_variable_list)]
                ShiftedPredictionData["Fcast"] = model.predict(Xpredict)

                # add the model id as a column, so we have a key when we write this data to the DB
                ShiftedPredictionData["ModelID"] = modelid

                # Rename the dependent variable column to 'Actual' for easy graphing
                ShiftedPredictionData.rename(
                    inplace=True, columns={dependent_variable: "Actual"}
                )

                # Get rid of any future-year rows that are empty for whatever reason
                CleanedPredictionResult = ShiftedPredictionData[
                    ["Actual", "Fcast"]
                ].dropna()

                # Calc the mean squared error for the entire prediction set
                model_results["predict_mse"] = mean_squared_error(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                model_results["predict_mape"] = mean_absolute_percentage_error(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                model_results["predict_mae"] = mean_absolute_error(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                model_results["predict_rsquared"] = r2_score(
                    CleanedPredictionResult[["Actual"]],
                    CleanedPredictionResult[["Fcast"]],
                )
                model_results["prediction_set"] = ShiftedPredictionData
                model_results["independent_variable_list"] = independent_variable_list
                model_results["dependent_variable"] = dependent_variable
                model_results['modelid'] = modelid
                model_results['drop_months'] = drop_months
    return model_results


def get_target_rsquared(var, engine):
    # query = (
    #         "select top 1 Max(r_squared) as MaxRSquared from fcast_results where [Dependent Variable] like '%"+var+"%' and [Independent Variable Count] = 1"

    #     )
    # return  pd.read_sql_query(sql=text(query), con=engine.connect())["MaxRSquared"]
    with engine.connect() as my_connection:
        return my_connection.execute(
            text(
                "select top 1 Max(r_squared) as MaxRSquared from fcast_results where [Dependent Variable] like '%"
                + var
                + "%' and [Independent Variable Count] = 1"
            )
        ).scalar()


def get_target_mse(var, engine):
    # query = (
    #         "select top 1 Max(mse) as MinMSE from fcast_results where [Dependent Variable] like '%"+var+"%' and [Independent Variable Count] = 1"

    #     )
    # return  pd.read_sql_query(sql=text(query), con=engine.connect())["MinMSE"]

    with engine.connect() as my_connection:
        return my_connection.execute(
            text(
                "select top 1 Max(mse) as MinMSE from fcast_results where [Dependent Variable] like '%"
                + var
                + "%' and [Independent Variable Count] = 1"
            )
        ).scalar()


def get_last_modelid(engine):
    with engine.connect() as my_connection:
        return my_connection.execute(
            text(
                "select top 1 modelid from "
                + table_prefix
                + "_results order by modelid desc"
            )
        ).scalar()


def get_invependent_vars_from_dependent_var(var, engine):
    query = (
        "select distinct Variable from fcast_high_scoring_vars where [Dependent Variable] = '"
        + var
        + "'"
    )

    return pd.read_sql_query(sql=text(query), con=engine.connect())["Variable"]


def get_category_variables(engine):
    global VarCategories
    VarCategories = {}
    VarCategoryList = []
    query = "select distinct Category from fcast_base_data_unpivoted_tmp where ItemVarType = 'I' and Category not in ('Utilization', 'Commodities','PriceIndex','Construction NSA','Consumer','Demand Driver') order by Category asc"
    VarCategoryList = pd.read_sql_query(sql=text(query), con=engine.connect())[
        "Category"
    ]
    for Cat in VarCategoryList:
        query = (
            "select distinct ItemName from fcast_base_data_unpivoted_tmp where Category = '"
            + Cat
            + "' and ItemVarType = 'I' and ItemMetaType not like '%12mo avg%'"
        )
        VarCategories[Cat] = pd.read_sql_query(sql=text(query), con=engine.connect())[
            "ItemName"
        ]


    

# Set up the DB connection
midb = init_database()
# print("Database initialized!")

# Get all the data out of the DB
get_data(midb)
# print("data gotten!")
get_category_variables(midb)


# drop the tables (there's a switch inside the function to see if we actually drop)
manage_tables(midb)

if ManageTables != "Drop":
    modelid = get_last_modelid(midb)

for dependent_variable in dependent_vars:
    for minshift in [12, 6, 3]:
        current_shift_range = [s for s in shift_range if s >= minshift]
        starting_variable_list = []
        for Cat in VarCategories:
            mape_to_beat = 2
            for independent_variable in VarCategories[Cat]:
                current_variable_list = starting_variable_list + [independent_variable]
                current_results = process_var_list(
                    current_variable_list, dependent_variable, current_shift_range
                )
                if current_results["test_mape"] < mape_to_beat:
                    best_results = current_results
                    mape_to_beat = current_results["test_mape"]
                    winning_variable = independent_variable
            starting_variable_list.append(winning_variable)
            save_result_set(best_results)