import pandas as pd

# Set display options to show all columns
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('C:\\Users\\amanv\\Downloads\\Unemployment in India.csv')

# Group the data by "Region"
grouped = df.groupby('Region')

# Iterate through each group and display the data
for region, region_data in grouped:
    print(f"State: {region}")
    print(region_data)

    # Convert labor participation rate to a decimal
    labor_participation_rate_decimal = region_data[' Estimated Labour Participation Rate (%)'] / 100

    # Calculate Total Labor Force
    total_labor_force = region_data[' Estimated Employed'] / (1 - labor_participation_rate_decimal)

    # Calculate Total Unemployed
    total_unemployed = total_labor_force - region_data[' Estimated Employed']

    print(f"Total Labor Force: {total_labor_force.sum()}")
    print(f"Total Unemployed: {total_unemployed.sum()}")

    print('\n')  # Add a newline for separation between states

