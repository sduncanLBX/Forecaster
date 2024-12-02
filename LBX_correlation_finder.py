import pandas as pd
from sqlalchemy import create_engine, text
from operator import itemgetter
import itertools
import statsmodels.api as sm
from statsmodels.tools import add_constant
import time

engine = create_engine(
    "mssql://@lbx-sqlintel/LBX_Market_Intelligence?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)
query = "select * from fcast_base_data_pivoted  where year > 2009 order by Year desc, Month  desc"

df = pd.read_sql_query(sql=text(query), con=engine.connect())


dependent_vars = [

  "RetailUnits",
    "RetailUnits_12mo_avg",
    "RetailUnits_lagging_indicator",
    "RetailUnits_lagging_indicator_direction",
    "RetailUnits_leading_indicator",
    "RetailUnits_leading_indicator_direction",
    "RetailUnits_pct_change_from_previous",
    "RetailUnits_pct_change_from_previous_direction",
    "RetailUnitsDirt",
    "RetailUnitsDirt_12mo_avg",
    "RetailUnitsDirt_lagging_indicator",
    "RetailUnitsDirt_lagging_indicator_direction",
    "RetailUnitsDirt_leading_indicator",
    "RetailUnitsDirt_leading_indicator_direction",
    "RetailUnitsDirt_pct_change_from_previous",
    "RetailUnitsDirt_pct_change_from_previous_direction",
    "RetailUnitsDORF",
    "RetailUnitsDORF_12mo_avg",
    "RetailUnitsDORF_lagging_indicator",
    "RetailUnitsDORF_lagging_indicator_direction",
    "RetailUnitsDORF_leading_indicator",
    "RetailUnitsDORF_leading_indicator_direction",
    "RetailUnitsDORF_pct_change_from_previous",
    "RetailUnitsDORF_pct_change_from_previous_direction",
    "RetailUnitsForestry",
    "RetailUnitsForestry_12mo_avg",
    "RetailUnitsForestry_lagging_indicator",
    "RetailUnitsForestry_lagging_indicator_direction",
    "RetailUnitsForestry_leading_indicator",
    "RetailUnitsForestry_leading_indicator_direction",
    "RetailUnitsForestry_pct_change_from_previous",
    "RetailUnitsForestry_pct_change_from_previous_direction",
    "RetailUnitsGenCon",
    "RetailUnitsGenCon_12mo_avg", 

    "RetailUnitsGenCon_lagging_indicator",
    "RetailUnitsGenCon_lagging_indicator_direction",
    "RetailUnitsGenCon_leading_indicator",
    "RetailUnitsGenCon_leading_indicator_direction",
    "RetailUnitsGenCon_pct_change_from_previous",
    "RetailUnitsGenCon_pct_change_from_previous_direction",
    "RetailUnitsMaterialHandling",
    "RetailUnitsMaterialHandling_12mo_avg",
    "RetailUnitsMaterialHandling_lagging_indicator",
    "RetailUnitsMaterialHandling_lagging_indicator_direction",
    "RetailUnitsMaterialHandling_leading_indicator",
    "RetailUnitsMaterialHandling_leading_indicator_direction",
    "RetailUnitsMaterialHandling_pct_change_from_previous",
    "RetailUnitsMaterialHandling_pct_change_from_previous_direction",
    "RetailUnitsRental",
    "RetailUnitsRental_12mo_avg",
    "RetailUnitsRental_lagging_indicator",
    "RetailUnitsRental_lagging_indicator_direction",
    "RetailUnitsRental_leading_indicator",
    "RetailUnitsRental_leading_indicator_direction",
    "RetailUnitsRental_pct_change_from_previous",
    "RetailUnitsRental_pct_change_from_previous_direction",
]


independent_vars = [

    "ABINationalBillings",
    "ABINationalBillings_12mo_avg",
    "ABINationalBillings_lagging_indicator",
    "ABINationalBillings_lagging_indicator_direction",
    "ABINationalBillings_leading_indicator",
    "ABINationalBillings_leading_indicator_direction",
    "ABINationalBillings_pct_change_from_previous",
    "ABINationalBillings_pct_change_from_previous_direction",
    "EFFR",
    "EFFR_12mo_avg",
    "EFFR_lagging_indicator",
    "EFFR_lagging_indicator_direction",
    "EFFR_leading_indicator",
    "EFFR_leading_indicator_direction",
    "EFFR_pct_change_from_previous",
    "EFFR_pct_change_from_previous_direction",
    "HOUST",
    "HOUST_12mo_avg",
    "HOUST_lagging_indicator",
    "HOUST_lagging_indicator_direction",
    "HOUST_leading_indicator",
    "HOUST_leading_indicator_direction",
    "HOUST_pct_change_from_previous",
    "HOUST_pct_change_from_previous_direction",
    "HOUST1FNSA",
    "HOUST1FNSA_12mo_avg",
    "HOUST1FNSA_lagging_indicator",
    "HOUST1FNSA_lagging_indicator_direction",
    "HOUST1FNSA_leading_indicator",
    "HOUST1FNSA_leading_indicator_direction",
    "HOUST1FNSA_pct_change_from_previous",
    "HOUST1FNSA_pct_change_from_previous_direction",
    "HOUSTNSA",
    "HOUSTNSA_12mo_avg",
    "HOUSTNSA_lagging_indicator",
    "HOUSTNSA_lagging_indicator_direction",
    "HOUSTNSA_leading_indicator",
    "HOUSTNSA_leading_indicator_direction",
    "HOUSTNSA_pct_change_from_previous",
    "HOUSTNSA_pct_change_from_previous_direction",
    # "RetailUnits",
    # "RetailUnits_12mo_avg",
    # "RetailUnits_lagging_indicator",
    # "RetailUnits_lagging_indicator_direction",
    # "RetailUnits_leading_indicator",
    # "RetailUnits_leading_indicator_direction",
    # "RetailUnits_pct_change_from_previous",
    # "RetailUnits_pct_change_from_previous_direction",
    # "RetailUnitsDirt",
    # "RetailUnitsDirt_12mo_avg",
    # "RetailUnitsDirt_lagging_indicator",
    # "RetailUnitsDirt_lagging_indicator_direction",
    # "RetailUnitsDirt_leading_indicator",
    # "RetailUnitsDirt_leading_indicator_direction",
    # "RetailUnitsDirt_pct_change_from_previous",
    # "RetailUnitsDirt_pct_change_from_previous_direction",
    # "RetailUnitsDORF",
    # "RetailUnitsDORF_12mo_avg",
    # "RetailUnitsDORF_lagging_indicator",
    # "RetailUnitsDORF_lagging_indicator_direction",
    # "RetailUnitsDORF_leading_indicator",
    # "RetailUnitsDORF_leading_indicator_direction",
    # "RetailUnitsDORF_pct_change_from_previous",
    # "RetailUnitsDORF_pct_change_from_previous_direction",
    # "RetailUnitsForestry",
    # "RetailUnitsForestry_12mo_avg",
    # "RetailUnitsForestry_lagging_indicator",
    # "RetailUnitsForestry_lagging_indicator_direction",
    # "RetailUnitsForestry_leading_indicator",
    # "RetailUnitsForestry_leading_indicator_direction",
    # "RetailUnitsForestry_pct_change_from_previous",
    # "RetailUnitsForestry_pct_change_from_previous_direction",
    # "RetailUnitsGenCon",
    # "RetailUnitsGenCon_12mo_avg",
    # "RetailUnitsGenCon_lagging_indicator",
    # "RetailUnitsGenCon_lagging_indicator_direction",
    # "RetailUnitsGenCon_leading_indicator",
    # "RetailUnitsGenCon_leading_indicator_direction",
    # "RetailUnitsGenCon_pct_change_from_previous",
    # "RetailUnitsGenCon_pct_change_from_previous_direction",
    # "RetailUnitsMaterialHandling",
    # "RetailUnitsMaterialHandling_12mo_avg",
    # "RetailUnitsMaterialHandling_lagging_indicator",
    # "RetailUnitsMaterialHandling_lagging_indicator_direction",
    # "RetailUnitsMaterialHandling_leading_indicator",
    # "RetailUnitsMaterialHandling_leading_indicator_direction",
    # "RetailUnitsMaterialHandling_pct_change_from_previous",
    # "RetailUnitsMaterialHandling_pct_change_from_previous_direction",
    # "RetailUnitsRental",
    # "RetailUnitsRental_12mo_avg",
    # "RetailUnitsRental_lagging_indicator",
    # "RetailUnitsRental_lagging_indicator_direction",
    # "RetailUnitsRental_leading_indicator",
    # "RetailUnitsRental_leading_indicator_direction",
    # "RetailUnitsRental_pct_change_from_previous",
    # "RetailUnitsRental_pct_change_from_previous_direction",
    "TLRESCON",
    "TLRESCON_12mo_avg",
    "TLRESCON_lagging_indicator",
    "TLRESCON_lagging_indicator_direction",
    "TLRESCON_leading_indicator",
    "TLRESCON_leading_indicator_direction",
    "TLRESCON_pct_change_from_previous",
    "TLRESCON_pct_change_from_previous_direction",
    "TTLCON",
    "TTLCON_12mo_avg",
    "TTLCON_lagging_indicator",
    "TTLCON_lagging_indicator_direction",
    "TTLCON_leading_indicator",
    "TTLCON_leading_indicator_direction",
    "TTLCON_pct_change_from_previous",
    "TTLCON_pct_change_from_previous_direction",
    "UMCSENT",
    "UMCSENT_12mo_avg",
    "UMCSENT_lagging_indicator",
    "UMCSENT_lagging_indicator_direction",
    "UMCSENT_leading_indicator",
    "UMCSENT_leading_indicator_direction",
    "UMCSENT_pct_change_from_previous",
    "UMCSENT_pct_change_from_previous_direction",
    "WPS0811",
    "WPS0811_12mo_avg",
    "WPS0811_lagging_indicator",
    "WPS0811_lagging_indicator_direction",
    "WPS0811_leading_indicator",
    "WPS0811_leading_indicator_direction",
    "WPS0811_pct_change_from_previous",
    "WPS0811_pct_change_from_previous_direction",
    "WPU0811",
    "WPU0811_12mo_avg",
    "WPU0811_lagging_indicator",
    "WPU0811_lagging_indicator_direction",
    "WPU0811_leading_indicator",
    "WPU0811_leading_indicator_direction",
    "WPU0811_pct_change_from_previous",
    "WPU0811_pct_change_from_previous_direction",
]
saved_results = []
# independent_vars = ['TTLCON']
# dependent_vars = ['RetailUnits']
ndf = pd.DataFrame(saved_results)
with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS fcast_correlation_finder_results"))
    connection.commit()

for depvar in dependent_vars:
    for indepvar in independent_vars:
        print(f"Dep Var: {depvar}, Indep Var: {indepvar}")
        for shift in range(-24, 1, 1):
            shifted_df = df[['DATE', depvar, indepvar]].copy()
            shifted_df[indepvar]=shifted_df[indepvar].shift(shift)

            

            # Drop rows with NaN values due to shifting
            shifted_df = shifted_df.dropna()

            # print("New iteration:")
            X = add_constant(shifted_df[indepvar])
            y = shifted_df[depvar]

            # Fit the regression model
            model = sm.OLS(y, X)
            results = model.fit()

            # print(f'Dep Var: {depvar}, Indep Var: {indepvar}, R-squared: {results.rsquared}')
            # Store the results
            saved_results.append(
                {
                    "Dependent Variable": depvar,
                    "Independent Variable": indepvar,
                    "Shift": shift,
                    "r_squared": results.rsquared,
                }
            )
            # if shift == -4:
            #     with engine.connect() as connection:
            #         shifted_df.to_sql("fcast_correlation_finder_testdata", connection, if_exists='replace', index=False)

        ndf = pd.DataFrame(saved_results)
        with engine.connect() as connection:
            ndf.to_sql("fcast_correlation_finder_results", connection, if_exists='append', index=False)
            connection.commit()
        saved_results = []
   #time.sleep(1)
