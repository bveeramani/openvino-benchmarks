library(ggplot2)
source("scripts/R/load_data.R")

# Prevents creation of Rplots.pdf
pdf(NULL)

models = models[models$batch_size == 32 & models$requests == 2 & models$api == "sync" & models$device == "MYRIAD", ]

ggplot(models, aes(ops, latency, colour=name)) +
  geom_point() +
  ggtitle("Prediction Latency vs # of Operations") +
  xlab("# of Operations") +
  ylab("Prediction Latency") +
  theme_minimal() +
  theme(strip.text.x = element_text(size = 8, angle = 90), axis.text.x = element_text(size = 5))
ggsave("./images/operations_vs_latency.png", width=8, height=6)

models = models[!grepl("_|fc|conv", models$name), ]
ggplot(models, aes(ops, latency, colour=name)) +
  geom_point() +
  ggtitle("Prediction Latency vs # of Operations") +
  xlab("# of Operations") +
  ylab("Prediction Latency (s)") +
  theme_minimal() +
  theme(strip.text.x = element_text(size = 8, angle = 90), axis.text.x = element_text(size = 5))
ggsave("./images/operations_vs_latency2.png", width=8, height=6)
