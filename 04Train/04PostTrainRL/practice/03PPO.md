# 手把手实现PPO

> 草稿

## train_ppo

```python
import torch
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import Dataset
import json

model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj"
                    ],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,
                                                          reward_adapter="./reward_model",
                                                          peft_config=peft_config,
                                                          quantization_config=bnb_config
                                                          )
model.to("cuda")

items = []
with open("./data/queries.json", "r", encoding="utf8") as f:
    for line in f:
        items.append(json.loads(line))
queries_dataset = Dataset.from_list(items)


def collator(data):
    queries = []
    for item in data:
        queries.append(tokenizer(item["query"], return_tensors="pt")["input_ids"].squeeze().to("cuda"))
    return queries


ppo_config = PPOConfig(kl_penalty="full", ppo_epochs=3, batch_size=2, mini_batch_size=1)
ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=None, tokenizer=tokenizer, dataset=queries_dataset,
                         data_collator=collator)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 32,
}

for batch in ppo_trainer.dataloader:
    query_tensors = batch

    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False,  **generation_kwargs)
    scores = []
    for query, response in zip(query_tensors, response_tensors):
        input_ids = torch.concat([query, response], dim=0)
        input_ids = torch.unsqueeze(input_ids, dim=0)
        score = ppo_trainer.model.compute_reward_score(input_ids=input_ids)[0, -1, 0]
        scores.append(score)
    stats = ppo_trainer.step(query_tensors, response_tensors, scores)
ppo_trainer.save_pretrained("./rl_model")
```

## train_reward

```python
import torch
from datasets import Dataset
import json

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                           num_labels=1,
                                                           quantization_config=bnb_config)
model.config.pad_token_id = tokenizer.pad_token_id
peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj"
                    ],
    task_type=TaskType.SEQ_CLS,
    lora_alpha=16,
    lora_dropout=0.05
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

items = []
with open("./data/preference.json", "r", encoding="utf8") as f:
    for line in f:
        item = json.loads(line)
        items.append(item)

dataset = Dataset.from_list(items)


def process_func(example):
    chosen = example["question"] + example["chosen"]
    rejected = example["question"] + example["rejected"]

    tokenized_chosen = tokenizer(chosen)
    tokenized_rejected = tokenizer(rejected)

    new_example = {}
    new_example["input_ids_chosen"] = tokenized_chosen["input_ids"]
    new_example["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    new_example["input_ids_rejected"] = tokenized_rejected["input_ids"]
    new_example["attention_mask_rejected"] = tokenized_rejected["attention_mask"]
    return new_example


dataset = dataset.map(process_func, remove_columns=['question', 'chosen', 'rejected'])
print(dataset)

config = RewardConfig(output_dir="./reward_model")
config.num_train_epochs = 1
config.per_device_train_batch_size = 1

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset
)
trainer.train()
trainer.save_model("./reward_model")
```



## data

### preference

```json
{"question":"Python中的字典是什么？", "chosen":"Python中的字典是一种无序的可变容器，允许使用键-值对来存储数据。","rejected":"Python中的字典用于存储数据。"}
{"question":"什么是回归分析？", "chosen":"回归分析是一种统计方法，用于确定变量之间的关系，通常用于预测和模型拟合。","rejected":"回归分析是一种数据分析方法。"}
{"question":"如何优化机器学习模型？", "chosen":"优化机器学习模型可以通过调节超参数、选择合适的模型和特征工程来实现。","rejected":"优化模型可以提高性能。"}
{"question":"什么是机器学习？", "chosen":"机器学习是一种人工智能方法，通过算法和统计模型使计算机系统能够执行特定任务，而无需明确编程指令。","rejected":"机器学习是计算机科学的一个领域。"}
{"question":"Python的主要特点是什么？", "chosen":"Python的主要特点包括简洁的语法、强大的库支持以及跨平台的兼容性。","rejected":"Python是一种编程语言。"}
{"question":"如何进行数据清洗？", "chosen":"数据清洗通常包括处理缺失值、去除重复数据、标准化数据格式等步骤。","rejected":"数据清洗是数据处理的一部分。"}
{"question":"什么是数据库？", "chosen":"数据库是一个有组织的数据集合，允许高效的数据存储、检索和管理。","rejected":"数据库用于存储数据。"}
{"question":"深度学习和机器学习的区别是什么？", "chosen":"深度学习是机器学习的一个子集，主要通过神经网络处理数据，而传统机器学习算法不依赖于神经网络。","rejected":"深度学习是机器学习的一部分。"}
{"question":"如何使用Pandas加载CSV文件？", "chosen":"你可以使用Pandas的`read_csv()`函数来加载CSV文件，并将其转换为DataFrame。","rejected":"使用Pandas加载CSV文件。"}
{"question":"什么是自然语言处理？", "chosen":"自然语言处理是计算机科学与人工智能的一个子领域，专注于机器与人类语言的交互。","rejected":"自然语言处理是计算机科学的一部分。"}
```

### queries

```json
{"query": "请给出保持健康的三个方法。"}
{"query": "三原色是什么？"}
{"query": "请描述原子的结构。"}
{"query": "如何减少空气污染？"}
{"query": "描述一次你不得不做出困难决定的经历。"}
{"query": "天空为什么是蓝色的？"}
{"query": "解释为什么4/16等同于1/4？"}
{"query": "写一个关于主人公必须做出重要职业决定的第三人称叙述的短故事。"}
{"query": "请渲染一座房子的三维模型。"}
{"query": "朱利叶斯·凯撒是如何死亡的？"}
```

