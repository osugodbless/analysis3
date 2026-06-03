import pandas as pd
from tableone import TableOne
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

print("Loading pre-cleaned dataset...")
df = pd.read_csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# ==============================================================================
# 1. ON-THE-FLY DATA PREPARATION
# ==============================================================================
# Extract continuous numeric age to calculate Mean (SD)
df['Numeric_Age'] = df['Age recode with <1 year olds and 90+'].astype(str).str.extract(r'(\d+)').astype(float)

# Filter out the "Other/No Treatment" wildcard group so the table only compares active clinical pathways
df = df[df['Clean_Treatment'] != 'Other/No Treatment']

# ==============================================================================
# 2. DEFINING THE TABLE PARAMETERS
# ==============================================================================
# List ALL columns you want in the table (Both continuous and categorical)
columns_to_summarize = [
    'Numeric_Age',       
    'Clean_Age',         
    'Clean_Race', 
    'Clean_Income',
    'Clean_Rural_Urban', 
    'Clean_Stage',
]

# List ONLY the categorical variables (Explicitly leave 'Numeric_Age' out of this list!)
categorical_variables = [
    'Clean_Age', 
    'Clean_Race', 
    'Clean_Income',
    'Clean_Rural_Urban', 
    'Clean_Stage',
]

# Rename the variables so they look professional in the final CSV
rename_dictionary = {
    'Numeric_Age': 'Age at Diagnosis, Mean (SD)',
    'Clean_Age': 'Age Groups',
    'Clean_Race': 'Race/Ethnicity',
    'Clean_Income': 'Median Income Bracket',
    'Clean_Rural_Urban': 'Rural/Urban',
    'Clean_Stage': 'Cancer Stage at Diagnosis',
}

# Grouping by your new primary clinical variable: Treatment Modality
groupby_variable = 'Clean_Treatment'

# ==============================================================================
# 3. GENERATING THE TABLE
# ==============================================================================
print("Calculating statistics and generating Table 1...")

# Define the logical order for the categorical variables so they print beautifully
categorical_order = {
    'Clean_Age': ['20 - 39 years', '40 - 59 years', '60 - 79 years', '80+ years'],
    'Clean_Stage': ['Localized', 'Regional', 'Distant', 'Unknown/unstaged'],
    'Clean_Income': ['< $60,000', '$60,000 - $99,999', '$100,000+', 'Unknown/Missing']
}

table1 = TableOne(
    df, 
    columns=columns_to_summarize, 
    categorical=categorical_variables, 
    groupby=groupby_variable, 
    rename=rename_dictionary,
    order=categorical_order,
    pval=True, 
    overall=True
)

print("\n")
print(table1.tabulate(tablefmt="github"))

output_file = "Table1_Ovarian_Baseline_Characteristics.csv"
table1.to_csv(output_file)
print(f"\nSuccess! Highly readable Table 1 saved to {output_file}")




# ------ For True Overall Cohort Analysis ------

# import pandas as pd
# from tableone import TableOne
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning) 

# print("Loading pre-cleaned dataset...")
# df_all = pd.read_csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# # Extract continuous numeric age
# df_all['Numeric_Age'] = df_all['Age recode with <1 year olds and 90+'].astype(str).str.extract(r'(\d+)').astype(float)

# columns_to_summarize = ['Numeric_Age', 'Clean_Age', 'Clean_Race', 'Clean_Income', 'Clean_Rural_Urban', 'Clean_Stage']
# categorical_variables = ['Clean_Age', 'Clean_Race', 'Clean_Income', 'Clean_Rural_Urban', 'Clean_Stage']
# categorical_order = {
#     'Clean_Age': ['20 - 39 years', '40 - 59 years', '60 - 79 years', '80+ years'],
#     'Clean_Stage': ['Localized', 'Regional', 'Distant', 'Unknown/unstaged'],
#     'Clean_Income': ['< $60,000', '$60,000 - $99,999', '$100,000+', 'Unknown/Missing']
# }

# print(f"Calculating the TRUE OVERALL column (N = {len(df_all):,})...")

# # Notice we are NOT grouping by treatment here!
# table1_overall = TableOne(
#     df_all, 
#     columns=columns_to_summarize, 
#     categorical=categorical_variables, 
#     order=categorical_order,
#     overall=True
# )

# print("\n")
# print(table1_overall.tabulate(tablefmt="github"))
# table1_overall.to_csv("Table1_TRUE_OVERALL_Column.csv")
# print("Saved to Table1_TRUE_OVERALL_Column.csv")