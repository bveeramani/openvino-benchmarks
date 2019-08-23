library(ggplot2)
source("scripts/R/load_data.R")

# Prevents creation of Rplots.pdf
pdf(NULL)

models = models[models$requests == 2 & models$batch_size == 32 & models$device == "MYRIAD", ]
models = models[!grepl("_|fc|conv", models$name), ]

ggplot(models, aes(reorder(name, throughput), throughput, fill=api)) +
  geom_bar(stat="identity", position="dodge") +
  ggtitle("API vs Throughput") +
  xlab("Name") +
  ylab("Throughput (fps)") +
  scale_y_continuous(expand = c(0, 0)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 5))
ggsave("./images/api_vs_throughput.png", width=8, height=6)
