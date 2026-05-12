from datasets import load_dataset
from collections import Counter

dataset = load_dataset("allocine")

# print(dataset)
print(len(dataset["train"]))
print(len(dataset["validation"]))
print(len(dataset["test"]))
print(dataset["train"].column_names)


# labels = dataset["train"]["label"]
# print(Counter(labels))

# for i in range(5):
#     print(dataset["train"][i]["review"])
#     print("Label:", dataset["train"][i]["label"])
#     print("-" * 50)

train_texts = dataset["train"]["review"]
train_labels = dataset["train"]["label"]

val_texts = dataset["validation"]["review"]
val_labels = dataset["validation"]["label"]

print(type(val_labels))