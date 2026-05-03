#!/usr/bin/env Rscript
# Simple scDesign3 generator for MERFISH hypothalamus.
# Run: mamba run -n r_env Rscript scDesign3/data_generation.r

suppressPackageStartupMessages({
  library(scDesign3)
  library(zellkonverter)
  library(SingleCellExperiment)
})

fractions <- c(0, 0.1, 0.25, 0.5, 1.0)
seed <- 1L
n_cores <- 2L
out_subdir <- "generated_h5ad"

args_cf <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_cf, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else getwd()
repo_root <- normalizePath(file.path(script_dir, ".."))
ref_h5ad <- normalizePath(file.path(repo_root, "data", "h5ad", "merfish_hypothalamus.h5ad"))
out_dir <- file.path(script_dir, out_subdir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(ref_h5ad)) stop("Missing file: ", ref_h5ad, call. = FALSE)
sce <- zellkonverter::readH5AD(ref_h5ad, reader = "python")
if (!"counts" %in% assayNames(sce)) assay(sce, "counts") <- as.matrix(assay(sce, assayNames(sce)[1]))

# Step 2 coordinate check: use only obsm['spatial'] (as reducedDim('spatial')).
if (!("spatial" %in% reducedDimNames(sce))) {
  stop("Expected reducedDim('spatial') from obsm['spatial'], but it is missing.", call. = FALSE)
}
sp <- as.matrix(reducedDim(sce, "spatial"))
if (ncol(sp) < 2) {
  stop("reducedDim('spatial') must have at least 2 columns.", call. = FALSE)
}
xy_from_spatial <- sp[, 1:2, drop = FALSE]
colData(sce)$x <- xy_from_spatial[, 1]
colData(sce)$y <- xy_from_spatial[, 2]
reducedDim(sce, "spatial") <- xy_from_spatial

# Step 3: midline is median x.
x0 <- median(colData(sce)$x)
colData(sce)$d_midline <- abs(colData(sce)$x - x0)
message("Midline x0 = median(x) = ", signif(x0, 6))

run_one <- function(mu_formula, use_d_midline, this_seed) {
  set.seed(this_seed)
  other_cov <- if (use_d_midline) "d_midline" else NULL
  out <- scDesign3::scdesign3(
    sce = sce,
    assay_use = "counts",
    celltype = NULL,
    pseudotime = NULL,
    spatial = NULL,
    other_covariates = other_cov,
    mu_formula = mu_formula,
    sigma_formula = "1",
    family_use = "nb",
    n_cores = n_cores,
    correlation_function = "default",
    usebam = FALSE,
    corr_formula = "1",
    copula = "gaussian",
    fastmvn = FALSE,
    DT = TRUE,
    pseudo_obs = FALSE,
    family_set = c("gauss", "indep"),
    important_feature = "all",
    nonnegative = TRUE,
    return_model = FALSE,
    nonzerovar = FALSE,
    parallelization = "mcmapply",
    BPPARAM = NULL,
    trace = FALSE
  )
  # Assumption: different scDesign3 versions expose counts via one of these fields.
  for (nm in c("new_count", "newcount", "count_mat")) {
    if (!is.null(out[[nm]])) return(as.matrix(out[[nm]]))
  }
  if (is.matrix(out)) return(out)
  stop("Unable to find simulated counts in scdesign3() output; inspect object names(out).", call. = FALSE)
}

all_genes <- rownames(sce)
n_genes <- length(all_genes)

message("Running baseline (mu = 1) simulation...")
counts_base <- run_one(mu_formula = "1", use_d_midline = FALSE, this_seed = seed)

message("Running covariate (mu = s(d_midline)) simulation...")
counts_cov <- run_one(mu_formula = "s(d_midline, bs='cr', k=10)", use_d_midline = TRUE, this_seed = seed + 1000L)

for (f in fractions) {
  if (f < 0 || f > 1) stop("fraction must be in [0,1]: ", f, call. = FALSE)
  set.seed(seed + as.integer(round(f * 1e6)))
  n_cov <- floor(f * n_genes)
  genes_cov <- if (n_cov > 0) sample(all_genes, n_cov) else character(0)
  genes_base <- setdiff(all_genes, genes_cov)

  out_counts <- counts_base
  if (n_cov > 0) out_counts[genes_cov, ] <- counts_cov[genes_cov, ]
  out_counts <- out_counts[all_genes, , drop = FALSE]

  out_sce <- SingleCellExperiment(assays = list(counts = out_counts), colData = colData(sce))
  reducedDim(out_sce, "spatial") <- xy_from_spatial

  tag <- gsub("\\.", "p", formatC(f, format = "f", digits = 4))
  out_path <- file.path(out_dir, paste0("merfish_midline_frac_", tag, ".h5ad"))
  zellkonverter::writeH5AD(out_sce, out_path, compression = "gzip")
  message("Wrote: ", out_path)
}

message("Done. Files in: ", normalizePath(out_dir))
