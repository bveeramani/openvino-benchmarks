library(ggplot2)
source("scripts/R/load_data.R")

# Prevents creation of Rplots.pdf
pdf(NULL)

models = models[models$batch_size == 32 & models$requests == 2 & models$api == "async" & models$device == "MYRIAD", ]
models = models[!grepl("_|fc|conv", models$name), ]

ggplot(models, aes(reorder(name, ops), ops, colour=name)) +
  geom_col() +
  ggtitle("Model Profiles") +
  xlab("Name") +
  ylab("# of Operations") +
  theme_minimal() +
  theme(axis.text.x = element_text(hjust=1, size = 5, angle = 90))
ggsave("./images/operations.png", width=8, height=6)
