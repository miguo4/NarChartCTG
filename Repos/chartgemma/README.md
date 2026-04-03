## Inference
1. Complete the basic environment setups
```
conda activate chartllama
cd Repos/chartgemma
python3 testGemma_intent.py
```
2. Set prompt style for both *Acc+* and *NQA* tasks in `./Repos/utils.py`
3. Modify the default path of `CKPT_PATH` in `./Repos/{MODEL_NAME}/infer.py`
4. Reimplement the `load_model` and `model_gen` functions
5. The results are saved in `./Result/raw/{MODEL_NAME}.jsonl` by default
6. Prompt LLMs in `./Stat/gpt_filter.py` to extract number values in NQA task
7. Set the parameters in `./Stat/stat_all_metric.py` and the statistical results are saved in `./Stat/Paper_Table`