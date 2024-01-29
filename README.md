# Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch

We conduct experiments on both encoder- and decoder-based LMs.
* For encoder-based LMs, we choose bert-base-uncased and roberta-base as pre-trained backbones. Eight datasets from the GLUE benchmark are used, including CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, and RTE.
* For decoder-based LMs, we choose LLaMA, Llama 2, and Code Llama as pre-trained backbones. WizardLM, WizardMath, WizardCoder-Python, and Code Alpaca are used as fine-tuned models. 
We evaluate three tasks on five datasets: AlpacaEval (instruction-following), GSM8K and MATH (mathematical reasoning), and HumanEval and MBPP (code-generating).

We provide GSM8K, MATH, and MBPP datasets in ```math_code_data/``` folder, which are obtained from [WizardLM repository](https://github.com/nlpxucan/WizardLM). 
Other datasets can be automatically downloaded by our codes. For language models, you can download them either manually or by our codes.   

You can also modify the ```cache_dir``` in the ```utils/load_config.py``` file to specify your own path to save datasets and models.


## Model Merging Methods

We provide a well-coded implementation of five model merging methods in this repository, including 
[Average Merging](https://arxiv.org/abs/2203.05482), 
[Task Arithmetic](https://arxiv.org/abs/2212.04089), 
[Fisher Merging](https://arxiv.org/abs/2111.09832), 
[RegMean](https://arxiv.org/abs/2212.09849), and 
[TIES-Merging](https://arxiv.org/abs/2306.01708). 
We also combine the proposed DARE with the above methods to facilitate the merging performance.


## Environments

[PyTorch 2.0.1](https://pytorch.org/),
[transformers 4.33.1](https://huggingface.co/docs/transformers/index),
[datasets 2.13.1](https://huggingface.co/docs/datasets/index),
[vllm 0.1.4](https://github.com/vllm-project/vllm),
[human_eval](https://github.com/openai/human-eval),
[numpy](https://github.com/numpy/numpy), and
[tqdm](https://github.com/tqdm/tqdm).


## Executing Scripts for Encoder-based LMs
For encoder-based LMs, we first fine-tune them on the GLUE benchmark (support both single-task and multi-task settings), 
and then inference with them. We also provide scripts to merge encoder-based LMs with five model merging methods. 

### Scripts for Fine-Tuning on GLUE
* Example of fine-tuning *roberta-base* on *CoLA* dataset under single-task setting:
```{bash}
python train_plms_glue.py --language_model_name roberta-base --dataset_name cola --learning_rate 1e-5 --num_runs 5
```
* Example of fine-tuning *roberta-base* on *CoLA* and *RTE* datasets under multi-task setting:
```{bash}
python train_plms_glue.py --language_model_name roberta-base --dataset_name cola --multitask_training --auxiliary_dataset_name rte --learning_rate 1e-5 --num_runs 5
```

### Scripts for Inference with DARE and Other Variants
* Example of direct inference on *roberta-base* (drop rate 0.0):
```{bash}
python inference_plms_glue.py --language_model_name roberta-base --weight_mask_rate 0.0
```
* Example of inference on *roberta-base* with DARE (drop rate 0.9):
```{bash}
python inference_plms_glue.py --language_model_name roberta-base --weight_mask_rate 0.9 --use_weight_rescale
```
* Example of inference on *roberta-base* with DropOnly (drop rate 0.9):
```{bash}
python inference_plms_glue.py --language_model_name roberta-base --weight_mask_rate 0.9
```
* Example of inference on *roberta-base* with magnitude-based pruning (drop rate 0.9):
```{bash}
python inference_plms_glue.py --language_model_name roberta-base --weight_mask_rate 0.9 --mask_strategy magnitude
```
* Example of inference on *roberta-base* with masking fine-tuned parameters (drop rate 0.9):
```{bash}
python inference_plms_glue.py --language_model_name roberta-base --weight_mask_rate 0.9 --use_weight_rescale --weight_format finetuned_weight
```

### Scripts for Merging Models
* Example of merging pairwise fine-tuned *roberta-base* with Average Merging:
```{bash}
python merge_plms_glue.py --merging_method_name average_merging --language_model_name roberta-base
```
* Example of merging pairwise fine-tuned *roberta-base* with Fisher Merging:
```{bash}
python merge_plms_glue.py --merging_method_name fisher_merging --normalize_fisher_weight --language_model_name roberta-base
```
* Example of merging pairwise fine-tuned *roberta-base* with Average Merging and DARE:
```{bash}
python merge_plms_glue.py --merging_method_name mask_merging --use_weight_rescale --language_model_name roberta-base --mask_apply_method average_merging
```


## Executing Scripts for Decoder-based LMs
Since the decoder-based LMs we use have already been fine-tuned, they can be directly utilized for inference.
We also provide scripts to merge decoder-based LMs with two model merging methods (Average Merging and Task Arithmetic).

### Scripts for Inference with DARE and Other Variants
* Example of direct inference on *WizardMath-7B-V1.0* on *GSM8K* (drop rate 0.0):
```{bash}
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name WizardMath-7B-V1.0 --tensor_parallel_size 1 --weight_mask_rate 0.0
```
* Example of inference on *WizardMath-7B-V1.0* on *GSM8K* with DARE (drop rate 0.9):
```{bash}
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name WizardMath-7B-V1.0 --tensor_parallel_size 1 --weight_mask_rate 0.9 --use_weight_rescale
```
* Example of inference on *WizardMath-7B-V1.0* on *GSM8K* with DropOnly (drop rate 0.9):
```{bash}
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name WizardMath-7B-V1.0 --tensor_parallel_size 1 --weight_mask_rate 0.9
```
* Example of inference on *WizardMath-7B-V1.0* on *GSM8K* with magnitude-based pruning (drop rate 0.9):
```{bash}
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name WizardMath-7B-V1.0 --tensor_parallel_size 1 --weight_mask_rate 0.9 --mask_strategy magnitude
```
* Example of inference on *WizardMath-7B-V1.0* on *GSM8K* with masking fine-tuned parameters (drop rate 0.9):
```{bash}
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name WizardMath-7B-V1.0 --tensor_parallel_size 1 --weight_mask_rate 0.9 --use_weight_rescale --weight_format finetuned_weight
```

### Scripts for Merging Models
* Example of merging *WizardLM-13B-V1.2* and *WizardMath-13B-V1.0* with Average Merging:
```{bash}
python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name average_merging --tensor_parallel_size 1
```
* Example of merging *WizardLM-13B-V1.2* and *WizardMath-13B-V1.0* with Task Arithmetic:
```{bash}
python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name task_arithmetic --scaling_coefficient 1.0 --tensor_parallel_size 1
```
* Example of merging *WizardLM-13B-V1.2* and *WizardMath-13B-V1.0* with Average Merging and DARE (drop rate 0.2):
```{bash}
python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name mask_merging --use_weight_rescale --weight_mask_rate 0.2 --mask_apply_method average_merging --tensor_parallel_size 1
```

**Note 1**: When merging decoder-based LMs, the number of GPUs we should allocate is equals to num_models_to_merge * tensor_parallel_size.
For example, if we want to merge *WizardLM-13B-V1.2* and *WizardMath-13B-V1.0* with tensor_parallel_size == 1, then we should allocate 2 * 1 = 2 GPUs.

**Note 2**: If "AssertionError: data parallel group is already initialized" error is raised by vllm on your device, please try to run ```direct_inference_merged_llms_instruct_math_code.py``` with the corresponding setting.
For example, if this error occurs when merging *WizardLM-13B-V1.2* and *WizardMath-13B-V1.0* with Average Merging and DARE (drop rate 0.2), please run the following command to evaluate on instruct- or math-related task
```{bash}
python direct_inference_merged_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name mask_merging --use_weight_rescale --weight_mask_rate 0.2 --mask_apply_method average_merging --tensor_parallel_size 1 --evaluate_task instruct
python direct_inference_merged_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name mask_merging --use_weight_rescale --weight_mask_rate 0.2 --mask_apply_method average_merging --tensor_parallel_size 1 --evaluate_task math
```

### Evaluation Process for AlpacaEval, HumanEval and MBPP
For AlpacaEval, HumanEval and MBPP, our codes will store the generated files and please additionally run the following evaluation commands to get the final metrics.

* For AlpacaEval:
We use ```chatgpt_fn``` in [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to compute the win rate. Firstly, please see [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to install the environment.
Then, if you want to evaluate the generated *WizardLM-13B-V1.2_inference_mask_0.2_rescale_True.json* file, please run
```{bash}
alpaca_eval --model_outputs ./save_gen_instruct_responses_results/alpaca_eval/WizardLM-13B-V1.2_inference_mask_0.2_rescale_True.json --annotators_config chatgpt_fn --name WizardLM-13B-V1.2_inference_mask_0.2_rescale_True
```

* For HumanEval:
Firstly, please see [human-eval repository](https://github.com/openai/human-eval) to install the environment.
Then, if you want to evaluate the generated *WizardCoder-Python-13B-V1.0_inference_mask_0.2_rescale_True.jsonl* file, please run
```{bash}
evaluate_functional_correctness ./save_gen_codes_results/human_eval/WizardCoder-Python-13B-V1.0_inference_mask_0.2_rescale_True.jsonl
```

* For MBPP:
Firstly, please see [bigcode-evaluation-harness repository](https://github.com/bigcode-project/bigcode-evaluation-harness) to install the environment.
Then, if you want to evaluate the generated *WizardCoder-Python-13B-V1.0_inference_mask_0.2_rescale_True.jsonl* file, please run
```{bash}
accelerate launch ./bigcode-evaluation-harness/main.py --tasks mbpp --allow_code_execution --load_generations_path ./save_gen_codes_results/mbpp/WizardCoder-Python-13B-V1.0_inference_mask_0.2_rescale_True.jsonl
```
