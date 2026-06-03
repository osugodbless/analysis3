# ==============================================================================
# FINE-GRAY ANALYSIS FOR RACIAL/ETHNIC DISPARITIES
# ==============================================================================
library(survival)

cat("Loading Ovarian dataset...\n")
data <- read.csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out 'Other/Unknown' and the very small AIAN group for the HR model 
# to ensure the model converges and remains stable.
data <- subset(data, Clean_Race != "Other/Unknown" & Clean_Race != "American Indian / Alaska Native")

data$Surv_Obj_Status <- factor(data$Event_State, 
                               levels = c(0, 1, 2, 3), 
                               labels = c("Censored", "CV_Death", "Cancer_Death", "Other_Death"))

# Setting Non-Hispanic White as the baseline reference
data$Clean_Race <- factor(data$Clean_Race, 
                          levels = c("Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Asian / Pacific Islander"))

cat("Building Fine-Gray model for Race...\n")
fg_data <- finegray(Surv(Survival_Months_Adjusted, Surv_Obj_Status) ~ Clean_Race, 
                    data = data, 
                    etype = "CV_Death")

fg_model_race <- coxph(Surv(fgstart, fgstop, fgstatus) ~ Clean_Race, 
                       weight = fgwt, 
                       data = fg_data)

summary(fg_model_race)