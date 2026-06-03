# ==============================================================================
# FINE-GRAY COMPETING RISKS ANALYSIS FOR OVARIAN AGE DISPARITIES (100% COHORT)
# ==============================================================================

if (!require("survival")) install.packages("survival", repos="http://cran.us.r-project.org")
library(survival)

cat("Loading full 100% Ovarian dataset (N ~ 38,148)...\n")
# Point directly to the main cleaned file!
data <- read.csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out Unknowns
data <- subset(data, Clean_Age != "Unknown")

# Format the Event Status
data$Surv_Obj_Status <- factor(data$Event_State, 
                               levels = c(0, 1, 2, 3), 
                               labels = c("Censored", "CV_Death", "Cancer_Death", "Other_Death"))

# Format the Age Variable and Set the Baseline
data$Clean_Age <- factor(data$Clean_Age, 
                         levels = c("20 - 39 years", "40 - 59 years", "60 - 79 years", "80+ years"))

cat("Building the Fine-Gray weighted dataset...\n")
fg_data <- finegray(Surv(Survival_Months_Adjusted, Surv_Obj_Status) ~ Clean_Age, 
                    data = data, 
                    etype = "CV_Death")

cat("Fitting the Cox Proportional Hazards Model to the Fine-Gray data...\n")
fg_model <- coxph(Surv(fgstart, fgstop, fgstatus) ~ Clean_Age, 
                  weight = fgwt, 
                  data = fg_data)

cat("\n====================================================================\n")
cat("          FINE-GRAY SUBDISTRIBUTION HAZARD RATIOS (AGE)             \n")
cat("====================================================================\n")
summary(fg_model)