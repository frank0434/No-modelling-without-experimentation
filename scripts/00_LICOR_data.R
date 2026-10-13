# %%
# =============================================================================
# LICOR_2024.R
# Step-by-step working script for the 2024 field LI-COR 6800 data only.
#   Step 1: read data (from the hand-cleaned .xlsx exports)
#   Step 2: filter on leakage            (added after Step 1 is verified)
#   Step 3: visualisation                (added after Step 2 is verified)
#
#  2024 measurements date 2024-06-17, 2024-06-26, 2024-07-11
# =============================================================================

library(data.table)
library(readxl)
library(ggplot2)

# ---- CONFIGURATION ----------------------------------------------------------
OD        <- "C:/Users/liu283/OneDrive - Wageningen University & Research/ch2_Expt"
DIR_2024  <- file.path(OD, "Expt_WUR_24/data/LICOR6800/2024-06-17potato_soilwarming")
LAYOUT_24 <- file.path(OD, "Expt_WUR_24/2024 WUR heat trial layout.xlsx")  # PlotID -> Treatment

# 2025 field: TWO folders (24 June + 3 July onward). xlsx exports are corrupt
# (20/22 export all-zero A), so 2025 is read from the raw tab-delimited TEXT logs.
DIR_2025_A <- file.path(OD, "Expt_WUR_25/Data/LICOR6800/24-06-25")   # 24 June
DIR_2025_B <- file.path(OD, "Expt_WUR_25/Data/LICOR6800/07-25")      # 3 July onward
LAYOUT_25  <- file.path(OD, "Expt_WUR_25/2025 WUR heat trial layout.xlsx")  # PlotID -> Treatment

SAT_QIN <- 1200   # umol m-2 s-1; light at/above this treated as saturating for potato
# Wheeler RM, Fitzpatrick AH and Tibbitts TW (2019) Potatoes as a Crop for Space Life Support:
# Effect of CO2, Irradiance, and Photoperiod on Leaf Photosynthesis and Stomatal Conductance. 
# Front. Plant Sci. 10:1632. doi: 10.3389/fpls.2019.01632
colors  <- c("20" = "#2c7bb6", "25" = "#1a9850", "30" = "#fdae61", "35" = "#d7191c")

# ---- helpers ----------------------------------------------------------------

# safe column getter: returns the column if present, else a vector of NA
gc <- function(dt, nm) if (nm %in% names(dt)) dt[[nm]] else rep(NA, nrow(dt))

# Read a raw 6800 .xlsx into a data.table whose column names come from the
# LI-COR "obs" header row (the 3-row header is category / name / units; we take
# the name row and drop the units row). Returns NULL for empty/aborted exports.
read_raw_6800_xlsx <- function(f) {
  raw <- as.data.table(read_excel(f, sheet = 1, col_names = FALSE,
                                  col_types = "text", .name_repair = "minimal"))
  if (ncol(raw) == 0 || nrow(raw) == 0) return(NULL)
  col1 <- as.character(raw[[1]])
  hr <- which(col1 == "obs")[1]
  if (is.na(hr)) return(NULL)
  hdr <- as.character(unlist(raw[hr]))
  hdr[is.na(hdr)] <- ""
  dt <- raw[(hr + 2):.N]                         # skip name row + units row
  setnames(dt, make.unique(hdr))                 # LI-COR reuses names (TIME, ...)
  dt
}

# Pull the standard gas-exchange columns, coerce numeric, drop the blank/units
# rows, derive date/datetime.
base_6800 <- function(raw) {
  num <- function(x) suppressWarnings(as.numeric(x))
  d <- data.table(
    obs      = num(gc(raw, "obs")),
    date_raw = as.character(gc(raw, "date")),
    A        = num(gc(raw, "A")),
    Ci       = num(gc(raw, "Ci")),
    gsw      = num(gc(raw, "gsw")),
    Qin      = num(gc(raw, "Qin")),
    Qabs      = num(gc(raw, "Qabs")),
    convert = num(gc(raw, "convert")), # J/umol
    Tleaf    = num(gc(raw, "Tleaf")),
    tair     = num(gc(raw, "Tair")),
    txchg_sp = num(gc(raw, "Txchg_sp")),
    leak_pct = num(gc(raw, "LeakPct")),
    stable   = num(gc(raw, "Stable"))
  )
  d <- d[!is.na(obs)]                             # drop units row / stray text
  if (nrow(d) == 0) return(NULL)
  d[, datetime := as.POSIXct(date_raw, format = "%Y%m%d %H:%M:%S", tz = "UTC")]
  d[, date := as.Date(datetime)]
  d[, date_raw := NULL]
  d[]
}

# --- 2025 helpers: raw TEXT log reader + per-file standardiser ----------------
# 2025 logs are tab-delimited text with the same 3-row header as the xlsx
# (category / name / units). Detect the "obs" name row, drop the units row.
# Returns NULL for aborted/empty logs (e.g. 2025-07-31-1139, 1 KB, no data).
read_raw_6800_txt <- function(f) {
  ln <- readLines(f, warn = FALSE)
  hr <- grep("^obs\t", ln)[1]
  if (is.na(hr) || length(ln) < hr + 2) return(NULL)
  hdr <- strsplit(ln[hr], "\t")[[1]]
  dt <- tryCatch(fread(text = paste(ln[(hr + 2):length(ln)], collapse = "\n"),
                       sep = "\t", header = FALSE, fill = TRUE),
                 error = function(e) NULL)
  if (is.null(dt) || nrow(dt) == 0) return(NULL)
  setnames(dt, make.unique(hdr[seq_len(ncol(dt))]))
  dt
}

# meta25: PlotID -> canonical Treatment (AC / HTI / HB); Treatment_old = H0/H1/H2
meta25 <- as.data.table(read_excel(LAYOUT_25, sheet = "Metadata"))[, .(PlotID, Treatment)]

# read one 2025 log -> standard columns. date + PlotID come from the FILENAME
# (authoritative), Treatment is joined from meta25, tleaf_set is fixed 25 C
# (every 2025 file is a single-plot A/Q light curve at ~25 C leaf temp).
read25 <- function(f) {
  raw <- read_raw_6800_txt(f)
  if (is.null(raw)) return(NULL)
  d <- base_6800(raw)
  if (is.null(d)) return(NULL)
  b <- basename(f)
  d[, fn        := b]
  d[, date      := as.Date(substr(b, 1, 10))]
  d[, PlotID    := as.integer(sub(".*[pP]\\s*([0-9]+).*", "\\1", b))]
  d[meta25, on = "PlotID", Treatment := i.Treatment]
  d[, tleaf_set := 25]
  # 24 June: plot 1's warming had not started yet -> record as control (AC)
  d[date == as.Date("2025-06-24") & PlotID == 1, Treatment := "AC"]
  d[]
}

# compact QC plot for one file's light curve (colour = 6800 stability flag)
qc_plot <- function(p) ggplot(p, aes(Qin, A)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_hline(yintercept = 0, colour = "grey85") +
  geom_line() + geom_point(aes(colour = factor(stable)), size = 2) +
  labs(title = paste0(p$date[1], "  p", p$PlotID[1], "  (", p$Treatment[1], ")"),
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)", colour = "stable") +
  theme_bw()


# ---- STEP 1: read + MANUAL per-file leaf-temp assignment --------------------
# tleaf_set is assigned BY HAND per file (obs range/list). Neither rounding the
# measured Tleaf (overshoots: 32.6 at a 30 C set -> rounds to 35) nor Txchg_sp
# (the exchanger setpoint reads ~5-7 C below the leaf) gives the design level
# reliably, and the files mix designs. So: read one file, look at the plot,
# set the fcase, drop the NA / warm-up / bad rows.
message("Reading field 2024 (.xlsx) ...")
files24 <- list.files(DIR_2024, pattern = "\\.xlsx$", full.names = TRUE)

read_one <- function(i) {
  d <- base_6800(read_raw_6800_xlsx(files24[i]))
  d[, fn := basename(files24[i])][]
}

# %% file 1 -- 2024-06-17-1223  single light curve @ 25 C (filename says 'test')
d1 <- read_one(1)
d1[, tleaf_set := fcase(obs %in% 1:13, 25, default = NA_real_)]
ggplot(d1, aes(Qin, A)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_line() + geom_point(aes(colour = factor(stable)), size = 2) +
  labs(title = "1223  A/Q @25C  (colour = stable flag)",
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)") + theme_bw()
d1 <- d1[!is.na(tleaf_set) & stable != 0]  # drop warm-up rows + unstable rows
ggplot(d1, aes(Qabs, A)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_line() + geom_point(aes(colour = factor(stable)), size = 2) +
  labs(title = "1223  A/Q @25C  (colour = stable flag)",
       x = "Qabs (umol m-2 s-1)", y = "A (umol m-2 s-1)") + theme_bw()
# %% file 3 -- 2024-06-26-0944  clean 6-light x 4-temp grid
d3 <- read_one(3)
d3[, tleaf_set := fcase(obs %in% 1:6,   20,
                        obs %in% 7:12,  25,
                        obs %in% 13:18, 30,
                        obs %in% 19:24, 35,
                        default = NA_real_)]
ggplot(d3, aes(Qin, A, colour = factor(tleaf_set))) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_line() + geom_point(size = 2) +
  facet_wrap(~ tleaf_set) +
  scale_colour_manual(values = colors, name = "Tleaf set (C)") +
  labs(title = "0944  A/Q x leaf-temp grid",
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)") + theme_bw()
d3 <- d3[!is.na(tleaf_set) & stable != 0]

# %% file 4 -- 2024-07-11-0923  TREATMENT x TEMPERATURE @ saturating light (Qin~1500)
# NOT a light curve. 8 plots (the AC + HTI plots) were each measured at 25/30/35 C,
# in plot order -> each consecutive obs TRIPLET is one plot. So:
#   1. drop the extra/bad obs 19 & 26   -> 24 rows = 8 plots x 3 temps
#   2. map PlotID positionally (each=3) and temp (rep 25/30/35, 8) on that structure
#   3. join Treatment from the layout metadata
#   4. THEN filter leak (LeakPct == 0) and drop obs 3
# (order matters: positional plot/temp mapping must happen before the leak filter,
#  or the triplet alignment breaks.)
meta      <- as.data.table(read_excel(LAYOUT_24, sheet = "Metadata"))
plots_923 <- meta[Treatment %in% c("AC", "HTI")]        # the 8 plots measured in 0923

d4 <- read_one(4)
d4 <- d4[!obs %in% c(19, 26)]                            # -> 24 rows = 8 plots x 3 temps
d4[, PlotID    := rep(plots_923$PlotID, each = 3)]
d4[, tleaf_set := rep(c(25, 30, 35),   times = 8)]
d4[plots_923[, .(PlotID, Treatment)], on = "PlotID", Treatment := i.Treatment]
d4 <- d4[leak_pct == 0 & obs != 3]                       # drop leaky rows + obs 3

ggplot(d4, aes(tleaf_set, A, colour = Treatment)) +
  geom_point(size = 2, position = position_jitter(width = .08, height = 0)) +
  geom_boxplot(aes(group = interaction(Treatment, tleaf_set)), alpha = 0.2, outlier.shape = NA) +
  labs(title = "0923  A by treatment x leaf-temp @ saturating light",
       x = "Treatment", y = "A (umol m-2 s-1)") + theme_bw()

# %% combine the four manually-curated files + tag measurement design
d2024 <- rbindlist(list(d1, d3, d4), fill = TRUE)
d2024[, measurement_type := fifelse(fn %like% "0923",
                                    "AT_temp_response", "AQ_light_response")]
d2024[, experiment := "field_2024"]
d2024[, Season := year(date)]
d2024[Season == 2024, Treatment := ifelse(is.na(Treatment), "AC", Treatment)]
d2024 |> 
  ggplot(aes(Qin, A, colour = factor(tleaf_set), shape = Treatment)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_point(size = 2) +
  geom_line() +
  facet_wrap(~ date ) +
  scale_colour_manual(values = colors, name = "Tleaf set (C)") +
  labs(title = "field_2024 A/Q curves (leaf-temp grid)",
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)") +
  theme_bw()
d2024[tleaf_set == 25 & date == as.Date("2024-06-17")] |> 
  ggplot(aes(Qin, A, colour = fn)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_point(size = 2) +
  geom_line() +
  labs(title = "field_2024 A/Q curves @25C (single-plot light curve)",
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)") +
  theme_bw()

cat("\n==== rows per file x tleaf_set (after manual assignment) ====\n")
print(dcast(d2024, fn ~ tleaf_set, fun.aggregate = length, value.var = "A"))
cat("\n==== stability / leak by file ====\n")
print(d2024[, .(n = .N, n_unstable = sum(stable == 0),
                n_leak_gt5 = sum(leak_pct > 5, na.rm = TRUE)), by = fn])
##### 2025 
# 24/06/2025	measurements 	"Leon did LICOR6800 for plot 1, 2, 4, 6 and 7. plot 1 is still control at this moment 
#  log file p7 before turn off the machine. So we didn't have the xlsx file. but we do have the raw data"
# # 3/07/2025	measurements 	LICOR6800 on plot 2, 3, 4 and 6. Leon is helping 
# 4/07/2025	measurements 	LICOR6800 on plot 7, 9, 10 and 11. 
# Plot 9 might not have been clamped properly leading to below than expected Amax. 
# The LICOR6800 had a couple temperory warnings 'cooling Td limited'. 
# the Tair temp was between 0.5 - 1.5 degree higher than Tleaf. 
# The Tirga was 30 degrees 
# 11/07/2025	measurements 	Li6800 on plot 2 and 3. The assimilation rate was really low (6). 
# The stomata was closed throughout the day. 
# 17/07/2025	measurements 	Li6800 on H2 and H0, plot 1, 2, 4 and 5
# 24/07/2025	measurements 	Li6800 on H2 and H0, plot 1, 2
# 31/07/2025	measurements 	Li6800 on H2 and H0, plot 1, 2 and 4

# =============================================================================
# STEP 1 (2025): read each single-plot light curve, eyeball it, drop bad obs.
# Each cell: read25(file) -> qc_plot() to look -> optional obs drop -> store.
# The store is keyed by filename so re-running a cell overwrites (idempotent).
# tleaf_set is fixed 25 C; date/PlotID/Treatment are set inside read25().
# =============================================================================
d25 <- list()

# %% ---- 24 June (folder DIR_2025_A) ----------------------------------------
# NB: p7 log has no xlsx twin (machine turned off); plot 1 recorded as AC.
p <- read25(file.path(DIR_2025_A, "2025-06-24-1030_log-P1"));     qc_plot(p)
# p <- p[!obs %in% c()]
qc_plot(p[stable != 0])  # drop unstable rows
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_A, "2025-06-24-1159_logdata-p2")); qc_plot(p)
p <- p[stable != 0]  # drop unstable rows
p <- p[!obs %in% c(1)]  # drop warm-up rows
qc_plot(p)
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_A, "2025-06-24-1235_logdata-p4")); qc_plot(p)
p <- p[stable != 0]  # drop unstable rows
p <- p[!obs %in% c(1)]  # drop warm-up rows
qc_plot(p)
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_A, "2025-06-24-1354_logdata-p6")); qc_plot(p)
p <- p[stable != 0]; qc_plot(p)  # drop unstable rows
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_A, "2025-06-24-1505_logdata-p7")); qc_plot(p)
d25[[p$fn[1]]] <- p

# %% ---- 3 July (folder DIR_2025_B) -----------------------------------------
p <- read25(file.path(DIR_2025_B, "2025-07-03-0927_logdata p2")); qc_plot(p)
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-03-1013_logdata p3")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-03-1107_logdata p4")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-03-1227_logdata p6")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- 4 July -------------------------------------------------------------
# plot 9 may not have clamped properly -> below-expected Amax (eyeball!).
p <- read25(file.path(DIR_2025_B, "2025-07-04-0905_logdata p7"));  qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-04-1022_logdata p9"));  qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-04-1127_logdata p10")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-04-1229_logdata p11")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- 11 July ------------------------------------------------------------
# stomata closed all day, A ~ 6 -> whole day is low (keep, but note it).
p <- read25(file.path(DIR_2025_B, "2025-07-11-1141_logdata p3")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-11-1305_logdata p2")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- 17 July ------------------------------------------------------------
p <- read25(file.path(DIR_2025_B, "2025-07-17-1001_logdata p1")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-17-1128_logdata p2")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-17-1216_logdata p4")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-17-1351_logdata p5")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- 18 July (not in logbook notes, but data present) -------------------
p <- read25(file.path(DIR_2025_B, "2025-07-18-1102_logdata p9"));  qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-18-1207_logdata p12")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-18-1257_logdata p10")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- 24 July ------------------------------------------------------------
p <- read25(file.path(DIR_2025_B, "2025-07-24-1238_logdata p1")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-24-1318_logdata p2")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- 31 July (the 1139 p2 log is a 1 KB aborted file -> skipped) --------
p <- read25(file.path(DIR_2025_B, "2025-07-31-1049_logdata p1")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-31-1300_logdata P2")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

p <- read25(file.path(DIR_2025_B, "2025-07-31-1331_logdata p4")); qc_plot(p)
# p <- p[!obs %in% c()]
d25[[p$fn[1]]] <- p

# %% ---- combine 2025 + visualise -------------------------------------------
d2025 <- rbindlist(d25, fill = TRUE)
d2025[, measurement_type := "AQ_light_response"]
d2025[, experiment       := "field_2025"]
d2025[, Season           := year(date)]
d2025 <- d2025[date != "2025-07-11" & stable != 0] # drop the 11 July files (stomata closed all day, A ~ 6) from the main dataset

ggplot(d2025, aes(Qin, A, colour = Treatment, group = fn)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_hline(yintercept = 0, colour = "grey85") +
  geom_line() + geom_point(size = 1.4) +
  facet_wrap(~ date) +
  labs(title = "field_2025 A/Q light curves (one plot per file, ~25 C)",
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)") + theme_bw()
# 12/07/2025	management	heating start H2
cat("\n==== 2025 curves: rows per date x plot (treatment) ====\n")
print(d2025[, .(n = .N, trt = Treatment[1],
                Amax = round(max(A), 1)), by = .(date, PlotID)][order(date, PlotID)])
cat("\n==== 2025 stability / leak by file ====\n")
print(d2025[, .(n = .N, n_unstable = sum(stable == 0),
                n_leak_gt5 = sum(leak_pct > 5, na.rm = TRUE)), by = fn])

d_field  <- rbindlist(list(d2024, d2025))
d_field[, .(n = .N, n_unstable = sum(stable == 0),
                n_leak_gt5 = sum(leak_pct > 5, na.rm = TRUE)), by = .(Season, experiment)]
# check unstable points if they still in line with other stable ones 
d_field|> 
  ggplot(aes(Qin, A, colour = factor(tleaf_set), shape = Treatment)) +
  geom_vline(xintercept = SAT_QIN, linetype = 2, colour = "grey60") +
  geom_point(size = 2) +
  facet_wrap(~ date ) +
  scale_colour_manual(values = colors, name = "Tleaf set (C)") +
  labs(title = "field_2024/2025 A/Q curves (leaf-temp grid)",
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)") +
  theme_bw()
# %% ---- 2021 control light response (ti8) -> light saturation point ---------
# Pre-treatment control leaf (Suzanne Brouwer chamber expt, 2021-06-23, ~21 C),
# used to pin the saturating-light threshold. Raw 2021 files keep the 6800 export
# in the "Measurements" sheet (sheet 1 is a readme). The curve steps Qin down
# 2000->0 (obs 1-21) then back up 0->2000 (obs 22-40). The DESCENDING arm is the
# clean light response; the ascending arm lags (stomata still opening) so it is
# EXCLUDED from the fit.
LR21_TI8 <- file.path(OD, "Project_SB_2021/data/raw/2021-06-23-0815_lightresponse_ti8.xlsx")

raw <- as.data.table(read_excel(LR21_TI8, sheet = "Measurements", col_names = FALSE,
                                col_types = "text", .name_repair = "minimal"))
hr  <- which(as.character(raw[[1]]) == "obs")[1]
hdr <- as.character(unlist(raw[hr])); hdr[is.na(hdr)] <- ""
raw <- raw[(hr + 2):.N]; setnames(raw, make.unique(hdr))   # 2021 Bluestem: select by NAME
ti8 <- base_6800(raw)
ti8[, arm := ifelse(obs <= 21, "descending (clean)", "ascending (stomata opening)")]

# --- fit TWO light-response models to the descending arm (obs 1-21) ----------
dsc <- ti8[obs %in% 1:21]

# (1) non-rectangular hyperbola (Marshall & Biscoe 1980): extra convexity param theta
nrh <- function(Q, phi, Am, t, Rd)
  (phi * Q + Am - sqrt((phi * Q + Am)^2 - 4 * t * phi * Q * Am)) / (2 * t) - Rd
fit_nrh <- nls(A ~ nrh(Qin, phi, Am, t, Rd), data = dsc,
               start = list(phi = 0.05, Am = 29, t = 0.75, Rd = 1.3),
               control = nls.control(maxiter = 500))
cn <- coef(fit_nrh); netn <- cn[["Am"]] - cn[["Rd"]]

# (2) negative-exponential / WOFOST-SUCROS form: Agross = Amax(1 - exp(-alpha*Q/Amax))
#     constrained: Amax > 0, alpha > 0, R >= 0
expm <- function(Q, Amax, alpha, R) Amax * (1 - exp(-alpha * Q / Amax)) - R
fit_exp <- nls(A ~ expm(Qin, Amax, alpha, R), data = dsc,
               start = list(Amax = 29, alpha = 0.06, R = 1),
               algorithm = "port", lower = c(Amax = 1e-6, alpha = 1e-6, R = 0),
               control = nls.control(maxiter = 500))
ce <- coef(fit_exp); nete <- ce[["Amax"]] - ce[["R"]]

# light saturation point = Qin where fitted net A reaches a fraction of its plateau
fn_n  <- function(Q) nrh(Q, cn[["phi"]], cn[["Am"]], cn[["t"]], cn[["Rd"]])
fn_e  <- function(Q) expm(Q, ce[["Amax"]], ce[["alpha"]], ce[["R"]])
qsat  <- function(fun, plateau, frac) uniroot(function(Q) fun(Q) - frac * plateau, c(1, 4000))$root
q90n <- qsat(fn_n, netn, .90); q95n <- qsat(fn_n, netn, .95)
q90e <- qsat(fn_e, nete, .90); q95e <- qsat(fn_e, nete, .95)

cat(sprintf("\nti8 NRH : Amax=%.1f theta=%.2f phi=%.3f R=%.2f | net=%.1f  sat90=%.0f sat95=%.0f  RSS=%.2f\n",
            cn[["Am"]], cn[["t"]], cn[["phi"]], cn[["Rd"]], netn, q90n, q95n, sum(resid(fit_nrh)^2)))
cat(sprintf("ti8 EXP : Amax=%.1f            alpha=%.3f R=%.2f | net=%.1f  sat90=%.0f sat95=%.0f  RSS=%.2f\n",
            ce[["Amax"]], ce[["alpha"]], ce[["R"]], nete, q90e, q95e, sum(resid(fit_exp)^2)))

# overlay both fitted curves; points coloured by arm, curves distinguished by linetype
grid   <- seq(0, 2000, 5)
curves <- rbindlist(list(
  data.table(model = "non-rect. hyperbola",      Qin = grid, A = fn_n(grid)),
  data.table(model = "neg-exponential (WOFOST)", Qin = grid, A = fn_e(grid))
))
ggplot(ti8, aes(Qin, A)) +
  geom_hline(yintercept = 0, colour = "grey85") +
  geom_point(aes(colour = arm), size = 2.4) +
  geom_line(data = curves, aes(Qin, A, linetype = model), inherit.aes = FALSE, linewidth = 0.7) +
  geom_vline(xintercept = SAT_QIN, linetype = 3, colour = "grey40") +
  scale_colour_manual(values = c("descending (clean)"         = "#2c7bb6",
                                 "ascending (stomata opening)" = "grey70")) +
  labs(title = "ti8 control light response (2021-06-23, pre-treatment, ~21 C)",
       subtitle = sprintf("NRH: Amax=%.1f theta=%.2f R=%.1f sat95=%.0f  |  EXP(WOFOST): Amax=%.1f alpha=%.3f R=%.1f sat95=%.0f",
                          cn[["Am"]], cn[["t"]], cn[["Rd"]], q95n, ce[["Amax"]], ce[["alpha"]], ce[["R"]], q95e),
       x = "Qin (umol m-2 s-1)", y = "A (umol m-2 s-1)", colour = "arm", linetype = "model") +
  theme_bw() + theme(legend.position = "bottom")

# =============================================================================
# CONTROLLED 2021 (Suzanne Brouwer chamber expt) -- cleaned/merged files.
# Read the three cleaned files one by one, eyeball each, then coerce ti8 + all
# three onto ONE canonical schema (SCHEMA) so they append onto the field data.
# NB: the cleaned snapshot file dropped Tleaf (raw files keep it) -> Tleaf is NA
# for snapshot/respiration here; that is expected, not a read error.
# =============================================================================
DIR_2021P <- file.path(OD, "Project_SB_2021/data/processed")

# canonical column order shared by field + controlled (NA where a source lacks it)
SCHEMA <- c("experiment","measurement_type","fn","date","Season",
            "PlotID","plant_id","Treatment","leaf",
            "obs","A","Ci","gsw","Qin", "Qabs", "convert","Tleaf","tleaf_set",
            "tair","txchg_sp","leak_pct","stable","datetime",
            "growing_temp","measuring_temp","day_code","dark_box","Fv_Fm","remarks")

# add any missing SCHEMA columns as NA, then order to SCHEMA
to_schema <- function(d) {
  miss <- setdiff(SCHEMA, names(d))
  if (length(miss)) d[, (miss) := NA]
  setcolorder(d, SCHEMA)
  d[, ..SCHEMA]
}

# read one cleaned 2021 file: strip "$unit" header suffixes, rename to the
# schema names, tag experiment/measurement_type/date. Units already match field.
read_clean21 <- function(fname, sheet, mtype) {
  d <- as.data.table(read_excel(file.path(DIR_2021P, fname), sheet = sheet))
  setnames(d, sub("\\$.*", "", names(d)))                       # drop $unit suffixes
  ren <- c(Plant = "plant_id", Leaf = "leaf", Leaf_number = "leaf",
           Day_in_heat = "day_code", Day_of_heat = "day_code",
           Growing_temp = "growing_temp", Growing_T = "growing_temp",
           Measuring_T = "measuring_temp", "Dark_box?" = "dark_box",
           "Fv/Fm" = "Fv_Fm", Remarks = "remarks")
  have <- intersect(names(ren), names(d)); setnames(d, have, ren[have])
  d[, `:=`(experiment = "controlled_2021", measurement_type = mtype,
           fn = fname, date = as.Date(Date), Season = 2021L)]
  d[, plant_id  := as.character(plant_id)]
  d[, leaf      := as.character(leaf)]
  d[, Treatment := as.character(Treatment)]
  d
}

# %% ---- standardise ti8 (control light response) to the schema -------------
# keep every row; the descending/ascending arm QC label is preserved in `remarks`.
ti8_std <- copy(ti8)
ti8_std[, `:=`(experiment = "controlled_2021", measurement_type = "AQ_light_response",
               fn = basename(LR21_TI8), Season = 2021L,
               plant_id = "ti8", Treatment = NA_character_, remarks = arm)]
ti8_std <- to_schema(ti8_std)
# %% ---- aci: A/Ci curves ----------------------------------------------------
aci <- read_clean21("acidata_2021_cleaned_merged.xlsx", "data", "ACi")
aci[Treatment %in% c("1", "2", "4"), Treatment := paste0("T", Treatment)]  # 1/2/4 -> T1/T2/T4
ggplot(aci, aes(Ci, A, colour = Treatment, group = interaction(plant_id, leaf, date))) +
  geom_line(alpha = .5) + geom_point(size = 1.3) +
  labs(title = "controlled_2021 A/Ci curves",
       x = "Ci (umol mol-1)", y = "A (umol m-2 s-1)") + theme_bw()
aci <- aci[A > -1]
aci <- to_schema(aci)

# %% ---- snap: snapshot A at growth light ------------------------------------
snap <- read_clean21("snapdata_2021_cleaned_merged.xlsx", "data", "snapshot")
ggplot(snap, aes(Treatment, A, fill = factor(growing_temp))) +
  geom_boxplot(outlier.size = .8) +
  labs(title = "controlled_2021 snapshot A by treatment x growing temp (n=653)",
       y = "A (umol m-2 s-1)", fill = "growing T (C)") + theme_bw()
snap <- to_schema(snap)

# %% ---- resp: dark respiration ----------------------------------------------
resp <- read_clean21("respirationdata_2021_cleaned_merged.xlsx", "respiration", "respiration")
ggplot(resp, aes(Treatment, A, fill = factor(measuring_temp))) +
  geom_hline(yintercept = 0, colour = "grey80") +
  geom_boxplot(outlier.size = .8) +
  labs(title = "controlled_2021 respiration A by treatment x measuring temp (n=227)",
       y = "A (umol m-2 s-1)", fill = "measuring T (C)") + theme_bw()
resp <- to_schema(resp)

# %% ---- combine controlled 2021 + append onto the field data ---------------
d2021 <- rbindlist(list(ti8_std, aci, snap, resp), fill = TRUE)   # all share SCHEMA
d_all <- rbindlist(list(d_field, d2021), fill = TRUE)             # controlled appended to field

cat("\n==== controlled_2021 rows by measurement_type ====\n")
print(d2021[, .(n = .N), by = measurement_type])
cat("\n==== d_all: rows by experiment x measurement_type ====\n")
print(dcast(d_all, experiment ~ measurement_type, fun.aggregate = length, value.var = "A"))
cat("\nd_all: one shared structure with", ncol(d_all), "columns\n")
fwrite(d_all, "data/LICOR_field_controlled_2021_2024_2025.csv")
d_all[is.na(Qabs),.(fn)] |> unique()
# count data points for each treatment x season x measurement_type combination
d_all[, .(n = .N), by = .(measurement_type, PlotID)]
colnames(d_all)
