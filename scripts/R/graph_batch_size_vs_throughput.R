library(ggplot2)
source("scripts/R/load_data.R")

# Prevents creation of Rplots.pdf
pdf(NULL)

models = models[models$requests == 2 & models$api == "async" & models$device == "MYRIAD, ]
models = models[!grepl("_|fc|conv", models$name), ]

ggplot(models, aes(batch_size, throughput, colour=name)) +
  geom_point() +
  ggtitle("Batch Size vs Throughput") +
  xlab("Batch Size") +
  ylab("Throughput (fps)") +
  facet_grid(. ~ name) +
  theme_minimal() +
  theme(strip.text.x = element_text(size = 8, angle = 90), axis.text.x = element_text(size = 5))
ggsave("./images/batch_size_vs_throughput.png", width=8, height=6)
