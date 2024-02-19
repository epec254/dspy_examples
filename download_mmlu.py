from datasets import load_dataset
import json

configs = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


loaded_data = []

for config in configs:
    print("loading " + config)
    dataset = {"data": load_dataset("tasksource/mmlu", config), "subject": config}

    loaded_data.append(dataset)
    # subjects.append(config)

answer_keys = ["choice_a", "choice_b", "choice_c", "choice_d"]

processed_data = {
    "dev": [],
    "validation": [],
    "test": [],
}
for data in loaded_data:
    print("saving " + data["subject"])
    # break
    for split in data["data"].keys():
        # print(split)
        # print(data[split])
        for row in data["data"][split]:
            # print(row)
            example = {
                "subject": data["subject"],
                "question": row["question"],
                "choice_a": row["choices"][0],
                "choice_b": row["choices"][1],
                "choice_c": row["choices"][2],
                "choice_d": row["choices"][3],
                "answer": answer_keys[row["answer"]],
            }
            # print(example)
            processed_data[split].append(example)
            # break
        # break

for split in ["dev", "validation", "test"]:
    print(len(processed_data[split]))
    with open(f"mmlu_{split}.jsonl", "w") as f:
        for item in processed_data[split]:
            f.write(json.dumps(item) + "\n")

# print(processed_data)
