# Usage: Rscript run_morphogam.R <out_dir> [knn]
# Reads <out_dir>/counts.csv (spots x genes, header = gene names) and
# <out_dir>/coords.csv (columns x, y in native units). Writes the CurveFinder
# coordinates, MorphoGAM results, and fitted t-effects back to <out_dir>.
suppressPackageStartupMessages(library(MorphoGAM))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
knn <- if (length(args) >= 2) as.integer(args[2]) else 5L
set.seed(0)

counts <- as.matrix(read.csv(file.path(out_dir, "counts.csv"), check.names = FALSE))
xy <- as.matrix(read.csv(file.path(out_dir, "coords.csv")))[, c("x", "y")]
Y <- t(counts)
rownames(Y) <- colnames(counts)

fit <- CurveFinder(xy, knn = knn)
mgam <- MorphoGAM(Y, curve.fit = fit, design = y ~ s(t, bs = "cr"))

write.csv(fit$xyt, file.path(out_dir, "morphogam_xyt.csv"), row.names = FALSE)
res <- mgam$results
res$gene <- rownames(res)
write.csv(res, file.path(out_dir, "morphogam_results.csv"), row.names = FALSE)
write.csv(t(mgam$fxs.t), file.path(out_dir, "morphogam_fxs_t.csv"), row.names = FALSE)
cat("MorphoGAM done:", nrow(res), "genes\n")
