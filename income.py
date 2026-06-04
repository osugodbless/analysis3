import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
from lifelines import AalenJohansenFitter

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Tied event times were detected.*")

print("Loading cleaned OVARIAN cancer dataset...")
df = pd.read_csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out the "Unknown/Missing" category
df = df[df['Clean_Income'] != 'Unknown/Missing']

# Define the exact order for processing
income_brackets = ['$100,000+', '$60,000 - $99,999', '< $60,000']

# Color gradient (Cool for high income, Warm for low income)
colors = {
    '$100,000+': 'steelblue',
    '$60,000 - $99,999': 'darkorange',
    '< $60,000': 'darkred'
}

print("\nFitting Aalen-Johansen Models for Income Brackets...\n")

fitters = {}
robust_cutoffs = {}
max_incidence_decimal = 0

for income in income_brackets:
    cohort = df[df['Clean_Income'] == income]
    
    if len(cohort) == 0:
        continue
        
    print(f"--- {income} (N = {len(cohort):,}) ---")
    
    ajf = AalenJohansenFitter(calculate_variance=True)
    ajf.fit(cohort['Survival_Months_Adjusted'], cohort['Event_State'], event_of_interest=1)
    fitters[income] = ajf
    
    # Truncation Math (N >= 100 at-risk for stability)
    robust_times = ajf.event_table[ajf.event_table['at_risk'] >= 100].index
    robust_cutoff = robust_times[-1] if len(robust_times) > 0 else ajf.cumulative_density_.index[-1]
    robust_cutoffs[income] = robust_cutoff
    
    # Milestone logic with the "NR" (Not Reached) safety check
    target_col = ajf.cumulative_density_.columns[0]
    def get_risk(month, cutoff):
        if month > cutoff:
            return "NR"
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
    
    print(f"  Final at Month {robust_cutoff:.1f}: {(final_est * 100):.2f}% (95% CI: [{(ci_lower * 100):.2f}% - {(ci_upper * 100):.2f}%])\n")
    
    if final_est > max_incidence_decimal:
        max_incidence_decimal = final_est

# Plotting
plt.figure(figsize=(10, 6))

for income in income_brackets:
    if income in fitters:
        ajf = fitters[income]
        cutoff = robust_cutoffs[income]
        plot_df = ajf.cumulative_density_[ajf.cumulative_density_.index <= cutoff]
        # Maintained linewidth of 1.5
        plt.plot(plot_df.index, plot_df.iloc[:, 0], label=income, color=colors[income], linewidth=1.5)

# Dynamic Scaling
upper_limit_percentage = (int((max_incidence_decimal * 100) / 5) + 1) * 5
plt.ylim(0, upper_limit_percentage / 100.0)
plt.xlim(0, 240)

plt.xlabel("Survival Months (Starting from Month 60)", fontsize=12)
plt.ylabel("Cumulative Incidence of CV Death (%)", fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, axis='y', alpha=0.3)
plt.legend(title="Median Household Income", loc='upper left')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("Ovarian_Disparities_Income_CV_Mortality.png", dpi=300)
print("Graph saved successfully as Ovarian_Disparities_Income_CV_Mortality.png!")
plt.show()