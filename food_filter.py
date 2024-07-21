import pandas as pd

# Load the entire CPI dataset
file_path = 'isp_data/data/Consumer Price Index by product group, monthly, percentage change, not seasonally adjusted, Canada, provinces, Whitehorse, Yellowknife and Iqaluit/18100004.csv'
cpi_df = pd.read_csv(file_path)

# Filter rows to only those where 'Products and product groups' column contains 'Food'
filtered_df = cpi_df[cpi_df['Products and product groups'] == 'Food']

# Save the filtered data to a new CSV file
filtered_df.to_csv('isp_data/data/Consumer Price Index by product group, monthly, percentage change, not seasonally adjusted, Canada, provinces, Whitehorse, Yellowknife and Iqaluit/filtered_18100004.csv', index=False)