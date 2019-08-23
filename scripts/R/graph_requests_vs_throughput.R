library(ggplot2)
source("scripts/R/load_data.R")

# Prevents creation of Rplots.pdf
pdf(NULL)

models = models[models$batch_size == 32 & models$api == "async" & models$device == "MYRIAD", ]
models = models[!grepl("_|fc|conv", models$name), ]

ggplot(models, aes(requests, throughput, colour=name)) +
  geom_point() +
  ggtitle("Number of Inference Requests vs Throughput") +
  xlab("# of Inference Requests") +
  ylab("Throughput (fps)") +
  facet_grid(. ~ name) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_minimal() +
  theme(strip.text.x = element_text(size = 8, angle = 90))
ggsave("./images/requests_vs_throughput.png", width=8, height=6)
