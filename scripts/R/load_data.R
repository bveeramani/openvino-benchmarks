library(tidyverse)

benchmarks = read.csv("data/benchmarks.csv", colClasses = c("factor", "integer", "integer", "numeric", "numeric", "factor", "factor"), na.strings = "None", stringsAsFactors = FALSE)

precision = benchmarks$device
levels(precision) = c("FP32", "FP16")
benchmarks$precision = precision

profiles = read.csv("data/profiles.csv", colClasses = c("factor", "factor", "integer", "integer", "integer", "numeric", "integer", "numeric"), na.strings = "None", stringsAsFactors = FALSE)
models = merge(benchmarks, profiles, by=c("name", "precision"))
models = models %>% distinct(name, batch_size, requests, device, api, .keep_all = TRUE)
