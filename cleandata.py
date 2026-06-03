import pandas as pd
import glob
import numpy as np
import re

print("Loading SEER data... (This may take a minute for millions of rows)")

# 1. Load and combine all your SEER CSV files
file_list = glob.glob("../raw_data/*.csv") # Ensure your SEER data files are in raw_data/
df_list = [pd.read_csv(file) for file in file_list]
df = pd.concat(df_list, ignore_index=True)

print(f"Total raw records loaded: {len(df):,}")

# ==============================================================================
# STEP 1: CRITICAL COHORT FILTERING (The "5-Year Ovarian" Cohort)
# ==============================================================================
print("Applying initial clinical filters...")

# 1. Filter for Ovarian Cancer ONLY
site_text = df['Site recode ICD-O-3/WHO 2008'].astype(str).str.lower()
df = df[site_text.str.contains('ovary', na=False)]

# 2. Filter for Females ONLY (Redundant for ovarian, but good for data integrity)
df = df[df['Sex'] == 'Female']

print(f"Total ovarian cancer patients: {len(df):,}")

# 3. Filter for >= 60 Survival Months
df['Survival months'] = pd.to_numeric(df['Survival months'], errors='coerce')
df = df.dropna(subset=['Survival months'])
df = df[df['Survival months'] >= 60]

print(f"Total master cohort (>5yr ovarian survivors): {len(df):,}")

# ==============================================================================
# STEP 2: THE "TIME ZERO" RESET
# ==============================================================================
# Since everyone survived at least 60 months, Month 60 is our new "Day 1".
df['Survival_Months_Adjusted'] = df['Survival months'] - 60

# ==============================================================================
# STEP 3: APPLY CUSTOM CLINICAL RECODES
# ==============================================================================
print("Applying custom categorization functions...")

# --- A. Age ---
def categorize_age_broad(val_str):
    val_str = str(val_str)
    match = re.search(r'\d+', val_str)
    if not match: return 'Unknown'
    age = int(match.group())
    if age < 40: return '20 - 39 years'
    elif 40 <= age < 60: return '40 - 59 years'
    elif 60 <= age < 80: return '60 - 79 years'
    else: return '80+ years'

df['Clean_Age'] = df['Age recode with <1 year olds and 90+'].apply(categorize_age_broad)

# --- B. Income ---
def categorize_income(val):
    val_str = str(val)
    if 'Unknown' in val_str: return 'Unknown/Missing'
    low_income = ['< $40,000', '$40,000 - $44,999', '$45,000 - $49,999', '$50,000 - $54,999', '$55,000 - $59,999']
    high_income = ['$100,000 - $109,999', '$110,000 - $119,999', '$120,000+']
    if val_str in low_income: return '< $60,000'
    elif val_str in high_income: return '$100,000+'
    else: return '$60,000 - $99,999'

df['Clean_Income'] = df['Median household income inflation adj to 2023'].apply(categorize_income)

# --- C. Race/Ethnicity ---
def categorize_race(val):
    val_str = str(val).strip()
    if val_str == 'Non-Hispanic White': return 'Non-Hispanic White'
    elif val_str == 'Non-Hispanic Black': return 'Non-Hispanic Black'
    elif val_str == 'Hispanic (All Races)': return 'Hispanic'
    elif val_str == 'Non-Hispanic Asian or Pacific Islander': return 'Asian / Pacific Islander'
    elif val_str == 'Non-Hispanic American Indian/Alaska Native': return 'American Indian / Alaska Native'
    else: return 'Other/Unknown'

df['Clean_Race'] = df['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'].apply(categorize_race)

# --- D. Stage ---
stage_col = 'Summary stage 2000 (1998-2017)' 
df['Clean_Stage'] = df[stage_col].replace({
    'Blank(s)': 'Unknown/unstaged',
    'Unknown/unstaged': 'Unknown/unstaged'
})

# --- E. Rural/Urban ---
def categorize_rural_urban(val):
    val_str = str(val).lower()
    if 'not adjacent' in val_str or 'nonmetropolitan' in val_str: return 'Rural (Nonmetropolitan)'
    elif 'metropolitan' in val_str: return 'Urban (Metropolitan)'
    else: return 'Unknown/Missing'

df['Clean_Rural_Urban'] = df['Rural-Urban Continuum Code'].apply(categorize_rural_urban)

# --- F. Surgery Status ---
def categorize_surgery(val):
    try:
        val_int = int(float(val))
    except (ValueError, TypeError):
        return 'Unknown'
    if val_int == 0: return 'No'
    elif 1 <= val_int <= 90: return 'Yes'
    else: return 'Unknown'

df['Surgery_Status'] = df['RX Summ--Surg Prim Site (1998+)'].apply(categorize_surgery)

# --- G. Chemotherapy ---
def categorize_chemo(val):
    val_str = str(val).strip()
    if val_str == 'Yes': return 'Yes'
    else: return 'No/Unknown'

df['Chemo_Status'] = df['Chemotherapy recode (yes, no/unk)'].apply(categorize_chemo)

# --- H. Cancer Treatment (Ovarian Specific) ---
def categorize_treatment(row):
    surg = str(row.get('Surgery_Status', '')).strip()
    chemo = str(row.get('Chemo_Status', '')).strip()
    
    # Ovarian cancer standard of care pathways
    if surg == 'Yes' and chemo == 'Yes':
        return 'Surgery + Chemotherapy'
    elif surg == 'Yes' and chemo == 'No/Unknown':
        return 'Surgery Only'
    elif surg != 'Yes' and chemo == 'Yes':
        return 'Chemotherapy Alone (No Surgery)'
    else:
        # Catches untreated patients or rare palliative radiation cases
        return 'Other/No Treatment'

df['Clean_Treatment'] = df.apply(categorize_treatment, axis=1)

# ==============================================================================
# STEP 4: DATA PREPARATION & MAPPING (Competing Risks)
# ==============================================================================
def define_competing_risk(row):
    status = str(row['Vital status recode (study cutoff used)']).strip().lower()
    cod = str(row['COD to site recode']).strip()
    
    if status == 'alive' or cod == 'Alive' or cod == 'State DC not available or state DC available but no COD':
        return 0 
    if cod == 'Diseases of Heart':
        return 1 
        
    other_cods = [
        'Accidents and Adverse Effects', 'Alzheimers (ICD-9 and 10 only)', 'Aortic Aneurysm and Dissection',                        
        'Atherosclerosis', 'Cerebrovascular Diseases', 'Certain Conditions Originating in Perinatal Period',
        'Chronic Liver Disease and Cirrhosis', 'Chronic Obstructive Pulmonary Disease and Allied Cond',
        'Complications of Pregnancy, Childbirth, Puerperium', 'Congenital Anomalies', 'Diabetes Mellitus',
        'Homicide and Legal Intervention', 'Hypertension without Heart Disease', 'In situ, benign or unknown behavior neoplasm', 
        'Nephritis, Nephrotic Syndrome and Nephrosis', 'Other Cause of Death', 'Other Diseases of Arteries, Arterioles, Capillaries',   
        'Other Infectious and Parasitic Diseases including HIV', 'Pneumonia and Influenza', 'Septicemia',
        'Stomach and Duodenal Ulcers', 'Suicide and Self-Inflicted Injury', 'Symptoms, Signs and Ill-Defined Conditions',
        'Syphilis', 'Tuberculosis'
    ]
    if cod in other_cods: return 3 
    return 2

df['Event_State'] = df.apply(define_competing_risk, axis=1)

# ==============================================================================
# STEP 5: EXPORT THE CLEANED COHORT
# ==============================================================================
# Note the updated filenames!
output_filename = "LongTerm_Ovarian_Survivors_Cleaned.csv"
print(f"Saving specifically targeted dataset to {output_filename}...")
df.to_csv(output_filename, index=False)

# print("Generating a 20% random sample for memory-safe R analysis...")
# df_sample = df.sample(frac=0.20, random_state=42) 

# sample_filename = "LongTerm_Ovarian_Survivors_For_R.csv"
# df_sample.to_csv(sample_filename, index=False)

print("\n" + "="*50)
print("OVARIAN CANCER DATA ENGINEERING COMPLETE!")
print("="*50)