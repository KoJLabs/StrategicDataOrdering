# Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning
This repository contains source code for our research on "[Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning](https://arxiv.org/pdf/2405.07490) version 2.0". 


## Data Ordering
Our experiments leverage a strategic data ordering process applied to several prominent large language models and datasets to demonstrate the efficacy of curriculum learning in natural language processing tasks.

### Data Ordering Techniques
In our experiments, we employ several data ordering techniques to arrange the training datasets optimally. The data ordering methods used are:

- Random: Shuffles the data randomly.
- Attention: Orders the data based on attention metrics derived from model.
- Loss: Prioritizes data samples according to the model’s prediction loss.
- Length: Sorts the data by the tokenized length of the prompt.

### Model Used
We utilized the following pre-trained models for our experiments:

- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [google/gemma-7b](https://huggingface.co/google/gemma-7b)

### Datasets
The data ordering process was applied to the following datasets:

- [Open-Orca/SlimOrca-Dedup](https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup)
- [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)


### Output Format
All datasets were reordered according to the curriculum learning strategy and are provided as Parquet files, which facilitate efficient handling of large datasets.

- {model_name}_{data_name}.parquet


## Getting Started
To run the experiments included in this study, follow these setup instructions.

### Prerequisites
Ensure you have Python 3.10 or newer installed on your system. You may also need to install additional Python libraries, which can be found in the pyproject.toml file:

```
poetry shell
poetry install
```

### Installation
Clone the repository to your local machine:

```
git clone https://github.com/KoJLabs/StrategicDataOrdering.git
cd StrategicDataOrdering
```


### Running the Preprocess Script
To start the data ordering process, use the following command:

```
python preprocess.py --model_path {model_path} --data_name {data_name} --max_length {max_length}
```


### Running the Training Script
To start the training process, use the following command:

```
python train.py --model_path {model_path} --data_name {data_name} --data_path {data_path} --lr {lr} --max_length {max_length} --save_path {save_path} --order_type {order_type} --epochs {epochs} --batch_size {batch_size}
```

## Citation
```
@misc{KoTAN,
  author       = {Juhwan Lee, Jisu Kim},
  title        = {Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning},
  howpublished = {\url{https://github.com/KoJLabs/StrategicDataOrdering}},
  year         = {2024},
}
```

```
@article{kim2024strategic,
  title={Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning},
  author={Kim, Jisu and Lee, Juhwan},
  journal={arXiv preprint arXiv:2405.07490},
  year={2024}
}
```
