import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
from lifelines import AalenJohansenFitter

# Suppress the expected tied-times warning to keep your terminal clean
warnings.filterwarnings("ignore", message=".*Tied event times were detected.*")

print("Loading cleaned >5-year OVARIAN cancer survivor dataset...")
df = pd.read_csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# ==============================================================================
# STEP 1: DEFINE COMPETING RISKS STATUS
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
    if cod in other_cods:
        return 3 
        
    return 2

print("Mapping competing risk events (This takes a moment)...")
# Safely reapplying just in case it wasn't preserved in a previous step
df['Event_State'] = df.apply(define_competing_risk, axis=1)

print("\n--- Event Frequencies ---")
print(f"Total Cohort: {len(df):,}")
print(f"Censored / Alive (Event 0): {len(df[df['Event_State'] == 0]):,}")
print(f"Strict Cardiac Deaths (Event 1): {len(df[df['Event_State'] == 1]):,}")
print(f"Cancer Deaths (Event 2): {len(df[df['Event_State'] == 2]):,}")
print(f"Other Non-Cancer Deaths (Event 3): {len(df[df['Event_State'] == 3]):,}")

# ==============================================================================
# STEP 2: FIT THE AALEN-JOHANSEN MODEL
# ==============================================================================
print("\nFitting the Aalen-Johansen Competing Risks Model...")

ajf = AalenJohansenFitter(calculate_variance=True)

ajf.fit(
    durations=df['Survival_Months_Adjusted'], 
    event_observed=df['Event_State'], 
    event_of_interest=1  
)

# ==============================================================================
# STEP 3: EXTRACT EXACT CLINICAL MILESTONES FOR YOUR PAPER
# ==============================================================================
ci_df = ajf.cumulative_density_ 
target_column = ci_df.columns[0] 

def get_risk_at_month(month, label):
    if month in ci_df.index:
        risk = ci_df.loc[month, target_column]
    else:
        valid_months = ci_df.index[ci_df.index <= month]
        if len(valid_months) == 0: return 0.0
        closest_month = valid_months[-1]
        risk = ci_df.loc[closest_month, target_column]
    
    print(f"Cumulative Incidence at {label}: {(risk * 100):.2f}%")

print("\n--- Cumulative Incidence Milestones ---")
get_risk_at_month(60, "Year 10 (60 months post-milestone)")
get_risk_at_month(120, "Year 15 (120 months post-milestone)")
get_risk_at_month(180, "Year 20 (180 months post-milestone)")

# ---------------------------------------------------------
# EXTRACT FINAL ESTIMATE (The "At-Risk Truncation" Method)
# ---------------------------------------------------------
event_table = ajf.event_table
robust_time_points = event_table[event_table['at_risk'] >= 100].index

if len(robust_time_points) > 0:
    true_final_time = robust_time_points[-1]
else:
    true_final_time = ajf.cumulative_density_.index[-1]

final_estimate = ajf.cumulative_density_.loc[true_final_time].iloc[0]
ci_lower = ajf.confidence_interval_.loc[true_final_time].iloc[0]
ci_upper = ajf.confidence_interval_.loc[true_final_time].iloc[1]
standard_error = (ci_upper - ci_lower) / 3.92

print(f"\n--- Overall Cohort (Truncated at N >= 100) ---")
print(f"Final Robust Time Point: Month {true_final_time}")
print(f"Final CVD Incidence: {(final_estimate * 100):.2f}%")
print(f"95% Confidence Interval: [{(ci_lower * 100):.2f}% - {(ci_upper * 100):.2f}%]")
print(f"Standard Error: {standard_error:.6f}\n")

# ==============================================================================
# STEP 4: PLOT THE SURVIVAL CURVE
# ==============================================================================
plt.figure(figsize=(10, 6))

ajf.plot(label='Cardiovascular Mortality Incidence (Overall Cohort)', color='darkred', linewidth=1.5)

# ---------------------------------------------------------
# DYNAMIC SCALING (Using our truncated estimate)
# ---------------------------------------------------------
plt.xlim(0, true_final_time)

max_percentage = final_estimate * 100 
upper_limit_percentage = (int(max_percentage / 5) + 1) * 5 
upper_limit_decimal = upper_limit_percentage / 100.0

plt.ylim(0, upper_limit_decimal)

plt.xlabel("Survival Months (Starting from Month 60)", fontsize=12)
plt.ylabel("Cumulative Incidence of CV Death (%)", fontsize=12)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.grid(True, axis='y', alpha=0.3)
plt.legend(loc='upper left')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

print("Generating visual plot...")
plt.savefig("Ovarian_Overall_CV_Mortality_Incidence.png", dpi=300)
print("Saved successfully to Ovarian_Overall_CV_Mortality_Incidence.png!")
plt.show()