# Capitalization and Punctuation for Automatic Speech Recognition

Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words. This repository provides code for training and predicting punctuation and capitalization for each word in a sentence to make ASR output more readable and to boost the performance of the named entity recognition, machine translation, or text-to-speech models. The model for this task was trained using a pre-trained BERT model. For every word in our training dataset, we’re going to predict:

- punctuation mark that should follow the word and
- whether the word should be capitalized

Main idea was introduced in the following paper with the official PyTorch implementation:
> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://arxiv.org/abs/2005.12592)<br>
> [Grammarly](https://github.com/grammarly/gector)

It is mainly based on `AllenNLP` and `transformers`.
## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

## Datasets
This model can work with any text dataset. The raw dataset should be preprocessed into two files, one `source` file and one `target` file. The `target` file should contain final texts, whereas the `source` file is simply the lowercase version with punctuations removed from the `target` file.<br>

**Note**: Punctuations should be space-separated with words.<br>

To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```

## Train model
To train the model, simply run:
```.bash
python train.py --train_set TRAIN_SET --dev_set DEV_SET \
                --model_dir MODEL_DIR
```
There are a lot of parameters to specify among them:
- `cold_steps_count` the number of epochs where we train only last linear layer
- `transformer_model {bert, distilbert, gpt2, roberta, transformerxl, xlnet, albert, xlm-r, phobert, ...}` model encoder
- `tn_prob` probability of getting sentences with no errors; helps to balance precision/recall
- `pieces_per_token` maximum number of subwords per token; helps not to get CUDA out of memory

## Model inference
To run your model on the input file use the following command:
```.bash
python predict.py --model_path MODEL_PATH [MODEL_PATH ...] \
                  --vocab_path VOCAB_PATH --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE
```
Among parameters:
- `min_error_probability` - minimum error probability (as in the paper)
- `additional_confidence` - confidence bias (as in the paper)
- `special_tokens_fix` to reproduce some reported results of pretrained models

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn  and
      Atrasevych, Vitaliy  and
      Chernodub, Artem  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA â†’ Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bea-1.16",
    pages = "163--170",
    abstract = "In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an F{\_}0.5 of 65.3/66.5 on CONLL-2014 (test) and F{\_}0.5 of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system.",
}
```
