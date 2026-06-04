# ==============================================================================
# FINE-GRAY ANALYSIS FOR RURAL/URBAN GEOGRAPHIC DISPARITIES
# ==============================================================================
library(survival)

cat("Loading Ovarian dataset...\n")
data <- read.csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out Unknowns
data <- subset(data, Clean_Rural_Urban != "Unknown/Missing")

# Format the Event Status
data$Surv_Obj_Status <- factor(data$Event_State, 
                               levels = c(0, 1, 2, 3), 
                               labels = c("Censored", "CV_Death", "Cancer_Death", "Other_Death"))

# Setting Urban as the baseline reference
data$Clean_Rural_Urban <- factor(data$Clean_Rural_Urban, 
                                 levels = c("Urban (Metropolitan)", "Rural (Nonmetropolitan)"))

cat("Building Fine-Gray model for Geography...\n")
fg_data <- finegray(Surv(Survival_Months_Adjusted, Surv_Obj_Status) ~ Clean_Rural_Urban, 
                    data = data, 
                    etype = "CV_Death")

fg_model_geo <- coxph(Surv(fgstart, fgstop, fgstatus) ~ Clean_Rural_Urban, 
                      weight = fgwt, 
                      data = fg_data)

summary(fg_model_geo)