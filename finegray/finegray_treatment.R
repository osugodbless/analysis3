# ==============================================================================
# FINE-GRAY ANALYSIS FOR TREATMENT MODALITY DISPARITIES
# ==============================================================================
library(survival)

cat("Loading Ovarian dataset...\n")
data <- read.csv("LongTerm_Ovarian_Survivors_Cleaned.csv")

# Filter out "Other/No Treatment" for the active analytic cohort
data <- subset(data, Clean_Treatment != "Other/No Treatment")

# Format the Event Status
data$Surv_Obj_Status <- factor(data$Event_State, 
                               levels = c(0, 1, 2, 3), 
                               labels = c("Censored", "CV_Death", "Cancer_Death", "Other_Death"))

# Setting Surgery Only as the baseline reference
data$Clean_Treatment <- factor(data$Clean_Treatment, 
                               levels = c("Surgery Only", "Surgery + Chemotherapy", "Chemotherapy Alone (No Surgery)"))

cat("Building Fine-Gray model for Treatment Modality...\n")
fg_data <- finegray(Surv(Survival_Months_Adjusted, Surv_Obj_Status) ~ Clean_Treatment, 
                    data = data, 
                    etype = "CV_Death")

fg_model_tx <- coxph(Surv(fgstart, fgstop, fgstatus) ~ Clean_Treatment, 
                     weight = fgwt, 
                     data = fg_data)

summary(fg_model_tx)