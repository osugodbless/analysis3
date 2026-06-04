# ==============================================================================
# FINE-GRAY ANALYSIS FOR CANCER STAGE DISPARITIES
# ==============================================================================
library(survival)

cat("Loading Ovarian dataset...\n")
data <- read.csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out Unknown/unstaged
data <- subset(data, Clean_Stage != "Unknown/unstaged")

# Format the Event Status
data$Surv_Obj_Status <- factor(data$Event_State, 
                               levels = c(0, 1, 2, 3), 
                               labels = c("Censored", "CV_Death", "Cancer_Death", "Other_Death"))

# Setting Localized as the baseline reference
data$Clean_Stage <- factor(data$Clean_Stage, 
                           levels = c("Localized", "Regional", "Distant"))

cat("Building Fine-Gray model for Stage...\n")
fg_data <- finegray(Surv(Survival_Months_Adjusted, Surv_Obj_Status) ~ Clean_Stage, 
                    data = data, 
                    etype = "CV_Death")

fg_model_stage <- coxph(Surv(fgstart, fgstop, fgstatus) ~ Clean_Stage, 
                        weight = fgwt, 
                        data = fg_data)

summary(fg_model_stage)