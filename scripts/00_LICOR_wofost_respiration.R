# %% Essential LICOR export: NLD2025 AMAX and supplementary NLD2024 temperature
# Controlled-2021 supplies measured respiration only; no A/Ci AMAX is exported.
# Raw compilation remains in 00_LICOR_data.R; historical fitting is archived.
library(data.table)
library(readxl)
licor <- fread("data/LICOR_field_controlled_2021_2024_2025.csv")
licor <- licor[experiment %in% c("field_2025", "field_2024", "controlled_2021")]
Q10 <- 2
UMOL_TO_KGHA <- 44.01e-9 * 1e4 * 3600
expm <- function(Qin, Amax_gross, alpha, Rd) {
  Amax_gross * (1 - exp(-alpha * Qin / Amax_gross)) - Rd
}

# %% Original measured respiration reference: grown/measured at 20 C
measured_dark <- licor[experiment == "controlled_2021" & measurement_type == "respiration"]
measured_dark[, Ci := as.integer(Ci)]
measured_dark <- measured_dark[Ci > 400 & Ci < 440]
measured_dark[, Rd := -A]
mean_Rd <- measured_dark[growing_temp == 20 & measuring_temp == 20, mean(Rd)]
T_mean_Rd <- 20
print(data.table(mean_Rd = mean_Rd, reference_temperature = T_mean_Rd, Q10 = Q10))

# %% WOFOST fits: NLD2025 main curves; NLD2024 AC curves for the temperature appendix
field_curves <- copy(licor[measurement_type == "AQ_light_response" &
                           (experiment == "field_2025" | (experiment == "field_2024" & Treatment == "AC"))])
field_curves[tleaf_set < 20, tleaf_set := round(Tleaf)]
curve_groups <- unique(field_curves[, .(fn, tleaf_set)])
print(unique(field_curves[, .(experiment, fn, tleaf_set)])[, .(n_curves = .N), by = experiment])
amax_rows <- list()
for (i in seq_len(nrow(curve_groups))) {
  obs_curve <- field_curves[fn == curve_groups$fn[i] & tleaf_set == curve_groups$tleaf_set[i]]
  temperature <- obs_curve$tleaf_set[1]
  Rd_fixed <- mean_Rd * Q10^((temperature - T_mean_Rd) / 10)
  curve_fit <- nls(
    A ~ expm(Qin, Amax_gross, alpha, Rd_fixed), data = obs_curve,
    start = list(Amax_gross = 29, alpha = 0.06), algorithm = "port",
    lower = c(Amax_gross = 1e-6, alpha = 1e-6),
    control = nls.control(maxiter = 500)
  )
  pars <- coef(curve_fit)
  amax_rows[[i]] <- data.table(
    source_id = paste(obs_curve$fn[1], temperature, sep = "_"),
    date = as.Date(obs_curve$date[1]), Treatment = if (obs_curve$experiment[1] == "field_2025") as.character(obs_curve$PlotID[1]) else obs_curve$Treatment[1],
    Tleaf = temperature, alpha = pars[["alpha"]],
    Amax = pars[["Amax_gross"]] - Rd_fixed, Rd = Rd_fixed,
    Gross_Amax = pars[["Amax_gross"]], cultivar = "Innovator",
    rss = sum(resid(curve_fit)^2), n = nrow(obs_curve), measurement_kind = "AQ_fit"
  )
}
# %% NLD2024 observed high-light readings are not fitted asymptotes
snapshot_amax <- copy(licor[
  experiment == "field_2024" & date == "2024-07-11" &
    measurement_type == "AT_temp_response" & Treatment == "AC"
])
snapshot_amax[, Rd := mean_Rd * Q10^((tleaf_set - T_mean_Rd) / 10)]
snapshot_amax <- snapshot_amax[, .(
  source_id = paste(fn, obs, sep = "_"), date = as.Date(date), Treatment,
  Tleaf = tleaf_set, Amax = A, Rd, Gross_Amax = A + Rd, cultivar = "Innovator",
  measurement_kind = "high_light"
)]

# %% Separate campaign exports; NLD2024 never enters the main AMAX table
amax_all <- rbindlist(list(rbindlist(amax_rows), snapshot_amax), fill = TRUE)
amax_all[, season := as.integer(format(date, "%Y"))]
emergence_dates <- data.table(season = c(2024L, 2025L),
                              emergence_date = as.Date(c("2024-06-07", "2025-05-22")))
amax_all <- emergence_dates[amax_all, on = "season"]
weather24 <- as.data.table(read_excel("data/Weather/WUR2024_wofost.xlsx", skip = 11))
weather25 <- as.data.table(read_excel("data/Weather/WUR2025_wofost.xlsx", skip = 11))
field_weather <- rbindlist(list(
  weather24[, .(season = 2024L, date = as.Date(date), Tmean = (Celsius...3 + Celsius...4) / 2)],
  weather25[, .(season = 2025L, date = as.Date(date), Tmean = (Celsius...3 + Celsius...4) / 2)]
))
amax_all[, gdd_emergence := NA_real_]
for (i in seq_len(nrow(amax_all))) {
  row <- amax_all[i]
  weather <- field_weather[season == row$season & date > row$emergence_date & date <= row$date]
  amax_all[i, gdd_emergence := sum(pmax(weather$Tmean - 2, 0))]
}
amax_all[, `:=`(
  site = "NLD", AMAX_wofost = Gross_Amax * UMOL_TO_KGHA, AMAX_net = Amax * UMOL_TO_KGHA,
  Tenv = Tleaf, amax_method = "wofost_fixed_mean_rd",
  mean_Rd = mean_Rd, Rd_reference_temperature = T_mean_Rd, Q10 = Q10
)]
fwrite(amax_all[season == 2025L], "output/licor_amax_wofost_fixed_rd.csv", encoding = "UTF-8")
fwrite(amax_all[season == 2024L & Treatment == "AC" & Tleaf >= 20 & gdd_emergence < 700],
        "output/licor_amax_leaf_temperature_NLD2024.csv", encoding = "UTF-8")
print(amax_all[, .(n = .N), by = .(season, measurement_kind)])
