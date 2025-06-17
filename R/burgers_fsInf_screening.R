gc()
rm(list = ls())

library(reticulate)
np <- import("numpy")

library(selectiveInference)

X_pre <- np$load("../Cache/X_pre_burgers_noise50.npy")
y_pre <- np$load("../Cache/y_pre_burgers_noise50.npy")
n = nrow(y_pre)

### data preprocessing ###
X_pre <- scale(X_pre)
y_pre <- scale(y_pre)

alpha = 0.05
sigmahat = sd(y_pre)
# run forward stepwise
fsfit = fs(X_pre,y_pre,intercept=FALSE,normalize=FALSE)
# run sequential inference with estimated sigma
out_fs = fsInf(fsfit,sigma=sigmahat,alpha=alpha,type="active")

save_file = "./R_data/fsInf_screening_burgers_noise50.rds"
saveRDS(out_fs, file = save_file)
readRDS(save_file)
