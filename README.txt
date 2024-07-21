This Independent Study Project will use Canadian food retail price data, Survey of household spending food expenditure data, household spending by household type, and related pricing data to generate sets of charts that illustrate how Canadian grocery spending has changed over time. It will adjust for inflation, and extend these trends to the future in attempt to predict where expenditures are heading.


Datasets used:


.isp_data/data/Monthly average retail prices for food/18100002.csv

Description: This Statistics Canada dataset contains grocery monthly average retail price sales data from January 1995 - February 2022. Key columns: REF_DATE, Products, UOM, VALUE, DECIMALS


./isp_data/data/Detailed food spending, Canada, regions and provinces/11100125.csv

Description: This Statistics Canada dataset contains Detailed food spending, for Canada by region and province from 2010-2021 on an annual basis. Key columns: REF_DATE, GEO, Food expenditures, summary-level categories, UOM, VALUE


./isp_data/data/Household spending by household type/11100224.csv

Description: This Statistics Canada dataset contains Household spending by household type data for Canada from 2010-2021 on an annual basis. This project will ignore non-grocery data within the set. Key columns: REF_DATE, Household type, Household expenditures summary-level categories, UOM, VALUE


./isp_data/data/Consumer Price Index by product group, monthly, percentage change, not seasonally adjusted, Canada, provinces, Whitehorse, Yellowknife and Iqaluit/18100004.csv

Description: Monthly Consumer Price Index by product group data from Statistics Canada separated by region. This project contains a script to filter a version of this dataset to include entries only for Food product categories, and uses it to consider inflation in historical price change analyses. The original full dataset filtered by this script has been removed due to filesize limitations so we will work with the filtered version. Key columns: REF_DATE, GEO, Products and product groups, VALUE


Other information:

food_filter.py is a short script to filter the large Canadian Consumer Price Index dataset to just food entries for the purpose of adjusting food prices for inflation in our overall analysis. It is used to create a filtered dataset placed in the "Consumer Price Index by product group..." data directory.

