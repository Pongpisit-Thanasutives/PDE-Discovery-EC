gc()
rm(list = ls())

library(reticulate)
np <- import("numpy")

library(selectiveInference)

X_pre <- np$load("../Cache/X_pre_kdv_noise50.npy")
y_pre <- np$load("../Cache/y_pre_kdv_noise50.npy")
n = nrow(y_pre)

alpha = 0.05
# run forward stepwise
fsfit = fs(X_pre,y_pre,intercept=FALSE,normalize=TRUE)
# run sequential inference with estimated sigma
out_fs = fsInf(fsfit,alpha=alpha,type="active")

save_file = "./R_data/fsInf_screening_kdv_noise50.rds"
saveRDS(out_fs, file = save_file)
readRDS(save_file)
