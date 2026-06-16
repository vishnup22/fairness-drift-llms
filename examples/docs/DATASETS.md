# Datasets

`data_loader.py` normalizes each benchmark into dictionaries with at least:

```text
prompt: text sent to the model
axis: demographic or benchmark axis when available
group: protected or target group when available
```

## Benchmarks

| Dataset | Loader | Source | Notes |
|---|---|---|---|
| BOLD | `load_bold` | `AlexaAI/bold` | Uses first prompt per row; includes category/domain metadata. |
| StereoSet | `load_stereoset` | `McGill-NLP/stereoset`, intersentence validation | Uses `context`; keeps stereotype and anti-stereotype fields. |
| BBQ | `load_bbq` | `heegyu/bbq`, Age test split | Combines context and question. |
| CrowS-Pairs | `load_crows_pairs` | local `data/crows_pairs_anonymized.csv` | Uses `sent_more`; local file must exist. |
| RealToxicityPrompts | `load_real_toxicity_prompts` | `allenai/real-toxicity-prompts` | Uses nested `prompt.text`. |
| HolisticBias | `load_holistic_bias` | `fairnlp/holistic-bias`, `sentences.csv` | Uses `sentence` or `text`. |
| WinoBias | `load_winobias` | `Elfsong/Wino_Bias` | Uses `input`; keeps gender/polarity metadata. |

## Adding a Benchmark

1. Add a loader in `data_loader.py`.
2. Return records through `_format_record`.
3. Include `axis` and `group` whenever available.
4. Add the loader to `load_all_datasets`.
5. Run a constrained smoke test:

```bash
python main.py --reuse-results --dataset=<existing_dataset>
python main.py --provider=<provider> --dataset=<new_dataset> --model=<small_model>
```
