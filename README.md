<div align="center">
<h1> Unleashing Large Language Models’ Proficiency in Zero-shot Essay Scoring
</center> <br> <center> </h1>

<p align="center">
Sanwoo Lee, Yida Cai, Desong Meng, Ziyang Wang, Yunfang Wu<sup>*</sup>
<br>
National Key Laboratory for Multimedia Information Processing, Peking University
<br>
<sup>*</sup> Corresponding author
<br><br>
EMNLP 2024 Findings <br>
<br>
</div>

This repository contains code for paper "Unleashing Large Language Models’ Proficiency in Zero-shot Essay Scoring", EMNLP 2024 Findings.

## Abstract
Advances in automated essay scoring (AES) have traditionally relied on labeled essays, requiring tremendous cost and expertise for their acquisition. Recently, large language models (LLMs) have achieved great success in various tasks, but their potential is less explored in AES. In this paper, we show that our zero-shot prompting framework, Multi Trait Specialization (MTS), elicits LLMs' ample potential for essay scoring. In particular, we automatically decompose writing proficiency into distinct traits and generate scoring criteria for each trait. Then, an LLM is prompted to extract trait scores from several conversational rounds, each round scoring one of the traits based on the scoring criteria. Finally, we derive the overall score via trait averaging and min-max scaling. Experimental results on two benchmark datasets demonstrate that MTS consistently outperforms straightforward prompting (Vanilla) in average QWK across all LLMs and datasets, with maximum gains of 0.437 on TOEFL11 and 0.355 on ASAP. Additionally, with the help of MTS, the small-sized Llama2-13b-chat substantially outperforms ChatGPT, facilitating an effective deployment in real applications.



## reproduce MTS for ASAP
```
git clone https://github.com/sanwooo/mts-llm-aes.git
cd mts-llm-aes/
```

**step 1.** Download and unzip the dataset from https://www.kaggle.com/competitions/asap-aes/data. Place `training_set_rel3.xlsx` under `ASAP/resource` folder. Also, please make sure that you manually fill out the content of the `prompt` column in `ASAP/resource/template_vanilla.xlsx` and  `ASAP/resource/template_mts.xlsx`. The contents for `prompt` column can be found in the `Essay_Set_Descriptions` folder of the downloaded dataset.

**step 2.** Run `ASAP/organize_dataset_and_template.py` to do basic preprocessing over the dataset. In particular, it will convert `@named_entity` into `{named_entity}` format and ensure the essay scores are aligned with the scoring rubric specified in `Essay_Set_Descriptions` folder. The resulting dataset `dataset.xlsx` will be placed in `ASAP/resource` folder.
```
python ASAP/organize_dataset_and_template.py
```
**step 3.** subsample the dataset and fill the content of essays into our templates. This will produce `df_test_{method}.xlsx` under `ASAP/resource` folder.
```
python ASAP/preprocess.py --method mts
```

**step 4.** inference with LLMs. This will produce `df_test_{model_name}_{method}.xlsx` under `ASAP/resource` folder.
```
python ASAP/inference_llama.py --method mts --model_name llama-2-7b-chat --batch_size 8
```
or
```
python ASAP/inference_mistral.py --method mts --model_name Mistral-7B-Instruct-v0.2 --batch_size 8
```

**step 5.** evaluate the scoring performance for the inference result produced by an LLM. This will print the QWKs for all prompts in the console.
```
python ASAP/postprocess.py --method mts --model_name {model_name}
```


## reproduce MTS for TOEFL11

**step 1.** Download the dataset from https://catalog.ldc.upenn.edu/LDC2014T06. Place `ETS_Corpus_of_Non-Native_Written_English` folder under `TOEFL11/resource` folder.

**step 2.** Run `TOEFL11/organize_dataset_and_template.py` to do basic preprocessing over the dataset. The resulting dataset `dataset.xlsx` will be placed in `TOEFL11/resource` folder.
```
python TOEFL11/organize_dataset_and_template.py
```
**step 3.** subsample the dataset and fill the content of essays into our templates. This will produce `df_test_{method}.xlsx` under `TOEFL11/resource` folder.
```
python TOEFL11/preprocess.py --method mts
```

**step 4.** inference with LLMs. This will produce `df_test_{model_name}_{method}.xlsx` under `TOEFL11/resource` folder.
```
python TOEFL11/inference_llama.py --method mts --model_name llama-2-7b-chat --batch_size 8
```
or
```
python TOEFL11/inference_mistral.py --method mts --model_name Mistral-7B-Instruct-v0.2 --batch_size 8
```

**step 5.** evaluate the scoring performance for the inference result produced by an LLM. This will print the QWKs for all prompts in the console.
```
python TOEFL11/postprocess.py --method mts --model_name {model_name}
```



