# ==============================================================================
# FINE-GRAY ANALYSIS FOR INCOME DISPARITIES
# ==============================================================================
library(survival)

cat("Loading Ovarian dataset...\n")
data <- read.csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out Unknowns
data <- subset(data, Clean_Income != "Unknown/Missing")

# Format the Event Status
data$Surv_Obj_Status <- factor(data$Event_State, 
                               levels = c(0, 1, 2, 3), 
                               labels = c("Censored", "CV_Death", "Cancer_Death", "Other_Death"))

# Setting high income as the baseline reference
data$Clean_Income <- factor(data$Clean_Income, 
                            levels = c("$100,000+", "$60,000 - $99,999", "< $60,000"))

cat("Building Fine-Gray model for Income...\n")
fg_data <- finegray(Surv(Survival_Months_Adjusted, Surv_Obj_Status) ~ Clean_Income, 
                    data = data, 
                    etype = "CV_Death")

fg_model_income <- coxph(Surv(fgstart, fgstop, fgstatus) ~ Clean_Income, 
                         weight = fgwt, 
                         data = fg_data)

summary(fg_model_income)