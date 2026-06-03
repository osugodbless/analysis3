import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
from lifelines import AalenJohansenFitter

warnings.filterwarnings("ignore", message=".*Tied event times were detected.*")

print("Loading cleaned >5-year OVARIAN cancer survivor dataset...")
df = pd.read_csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out missing age data
df = df[df['Clean_Age'] != 'Unknown']

# ==============================================================================
# STEP 1: DATA PREPARATION & MAPPING
# ==============================================================================
# The Event_State column is already mapped in your cleaned CSV, but if you need 
# to re-run it dynamically, it's safe to keep the mapping function here.
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

if 'Event_State' not in df.columns:
    df['Event_State'] = df.apply(define_competing_risk, axis=1)

# Define the exact order we want them processed and plotted
age_brackets = ['20 - 39 years', '40 - 59 years', '60 - 79 years', '80+ years']

# Define a color gradient (Cool to Warm)
colors = {
    '20 - 39 years': 'steelblue',
    '40 - 59 years': 'mediumseagreen',
    '60 - 79 years': 'darkorange',
    '80+ years': 'darkred'
}

# ==============================================================================
# STEP 2 & 3: FIT MODELS & EXTRACT MILESTONES
# ==============================================================================
print("\nFitting Aalen-Johansen Models for Age Brackets...\n")

fitters = {}
robust_cutoffs = {}
max_incidence_decimal = 0

for age in age_brackets:
    cohort = df[df['Clean_Age'] == age]
    
    if len(cohort) == 0:
        continue
        
    print(f"--- {age} (N = {len(cohort):,}) ---")
    
    ajf = AalenJohansenFitter(calculate_variance=True)
    ajf.fit(cohort['Survival_Months_Adjusted'], cohort['Event_State'], event_of_interest=1)
    fitters[age] = ajf
    
    # 1. Truncation Math (N >= 100)
    robust_times = ajf.event_table[ajf.event_table['at_risk'] >= 100].index
    robust_cutoff = robust_times[-1] if len(robust_times) > 0 else ajf.cumulative_density_.index[-1]
    robust_cutoffs[age] = robust_cutoff
    
    # Extract Milestones safely, obeying the robust cutoff!
    target_col = ajf.cumulative_density_.columns[0]
    def get_risk(month, cutoff):
        if month > cutoff:
            return "Nil"
        valid_months = ajf.cumulative_density_.index[ajf.cumulative_density_.index <= month]
        closest = valid_months[-1] if len(valid_months) > 0 else 0
        risk_val = ajf.cumulative_density_.loc[closest, target_col]
        return f"{(risk_val * 100):.2f}%"
    
    print(f"  Year 10 (60 mo):  {get_risk(60, robust_cutoff)}")
    print(f"  Year 15 (120 mo): {get_risk(120, robust_cutoff)}")
    print(f"  Year 20 (180 mo): {get_risk(180, robust_cutoff)}")
    
    # Final stable estimate
    final_est = ajf.cumulative_density_.loc[robust_cutoff].iloc[0]
    ci_lower = ajf.confidence_interval_.loc[robust_cutoff].iloc[0]
    ci_upper = ajf.confidence_interval_.loc[robust_cutoff].iloc[1]
    
    print(f"  Final at Month {robust_cutoff}: {(final_est * 100):.2f}% (95% CI: [{(ci_lower * 100):.2f}% - {(ci_upper * 100):.2f}%])\n")
    
    # Track the absolute highest incidence for dynamic Y-axis scaling
    if final_est > max_incidence_decimal:
        max_incidence_decimal = final_est

# ==============================================================================
# STEP 4: PLOT THE STRATIFIED CURVES
# ==============================================================================
plt.figure(figsize=(10, 6))

for age in age_brackets:
    if age in fitters:
        ajf = fitters[age]
        cutoff = robust_cutoffs[age]
        plot_df = ajf.cumulative_density_[ajf.cumulative_density_.index <= cutoff]
        plt.plot(plot_df.index, plot_df.iloc[:, 0], label=age, color=colors[age], linewidth=1.5)

# Dynamic Y-Axis Scaling
upper_limit_percentage = (int((max_incidence_decimal * 100) / 5) + 1) * 5
plt.ylim(0, upper_limit_percentage / 100.0)
plt.xlim(0, 240)

plt.xlabel("Survival Months (Starting from Month 60)", fontsize=12)
plt.ylabel("Cumulative Incidence of CV Death (%)", fontsize=12)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, axis='y', alpha=0.3)
plt.legend(title="Age at Diagnosis", loc='upper left')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

print("Generating visual plot...")
plt.savefig("Ovarian_Disparities_Age_CV_Mortality.png", dpi=300)
print("Saved successfully!")
plt.show()