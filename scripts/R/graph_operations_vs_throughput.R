library(ggplot2)
source("scripts/R/load_data.R")

# Prevents creation of Rplots.pdf
pdf(NULL)

models = models[models$batch_size == 32 & models$requests == 2 & models$api == "async", ]
models = models[!grepl("SqueezeNet", models$name), ]

ggplot(models, aes(ops, throughput, colour=device)) +
  geom_point() +
  ggtitle("# of Operations vs Throughput") +
  xlab("# of Operations") +
  ylab("Throughput (fps)") +
  theme_minimal() +
ggsave("./images/operations_vs_throughput.png", width=8, height=6)

ggplot(models[models$device == "MYRIAD", ], aes(ops, throughput, colour=name)) +
  geom_point() +
  ggtitle("# of Operations vs Throughput") +
  xlab("# of Operations") +
  ylab("Throughput (fps)") +
  theme_minimal() +
ggsave("./images/operations_vs_throughput_myriad.png", width=8, height=6)

ggplot(models[models$device == "CPU", ], aes(ops, throughput, colour=name)) +
  geom_point() +
  ggtitle("# of Operations vs Throughput") +
  xlab("# of Operations") +
  ylab("Throughput (fps)") +
  theme_minimal() +
ggsave("./images/operations_vs_throughput_cpu.png", width=8, height=6)

models = models[!grepl("_|fc|conv", models$name), ]
ggplot(models[models$device == "MYRIAD", ], aes(ops, throughput, colour=name)) +
  geom_point() +
  ggtitle("# of Operations vs Throughput") +
  xlab("# of Operations") +
  ylab("Throughput (fps)") +
  theme_minimal() +
ggsave("./images/operations_vs_throughput_myriad2.png", width=8, height=6)

ggplot(models[models$device == "CPU", ], aes(ops, throughput, colour=name)) +
  geom_point() +
  ggtitle("# of Operations vs Throughput") +
  xlab("# of Operations") +
  ylab("Throughput (fps)") +
  theme_minimal() +
ggsave("./images/operations_vs_throughput_cpu2.png", width=8, height=6)
