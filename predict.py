import argparse
from shlex import split

from utils.helpers import read_lines
from gector.gec_model import GecBERTModel
from tqdm import tqdm
import re


def predict_for_file(
    input_file, 
    output_file, 
    model, 
    batch_size=32, 
    split_chunk=False, 
    chunk_size=32, 
    overlap_size=8,
    min_words_cut=4
):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in tqdm(test_data):
        batch.append(sent.split())
        if len(batch) == batch_size:
            if split_chunk:
                batch, batch_indices = split_chunks(batch, chunk_size, overlap_size)
                preds, cnt = model.handle_batch(batch)
                preds = [
                    merge_chunk(preds[start:end], overlap_size, min_words_cut) 
                    for (start, end) in batch_indices
                ]
            else:
                preds, cnt = model.handle_batch(batch)
                preds = [" ".join(x) for x in preds]
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        if split_chunk:
            batch, batch_indices = split_chunks(batch, chunk_size, overlap_size)
            preds, cnt = model.handle_batch(batch)
            preds = [
                    merge_chunk(preds[start:end], overlap_size, min_words_cut) 
                    for (start, end) in batch_indices
            ]
        else:
            preds, cnt = model.handle_batch(batch)
            preds = [" ".join(x) for x in preds]
        predictions.extend(preds)
        cnt_corrections += cnt

    with open(output_file, 'w') as f:
        f.write("\n".join(predictions) + '\n')
    return cnt_corrections


def split_chunks(batch, chunk_size=32, overlap_size=8):
    # return batch pairs of indices
    stride = chunk_size - overlap_size
    result = []
    indices = []
    for tokens in batch:
        start = len(result)
        num_token = len(tokens)
        if num_token <= chunk_size:
            result.append(tokens)
        elif num_token > chunk_size and num_token < (chunk_size * 2 - overlap_size):
            split_idx = (num_token + overlap_size + 1) // 2
            result.append(tokens[:split_idx])
            result.append(tokens[split_idx - overlap_size:])
        else:
            for i in range(0, num_token - overlap_size, stride):
                result.append(tokens[i: i + chunk_size])

        indices.append((start, len(result)))

    return result, indices


def check_alnum(s):
    if len(s) < 2:
        return False
    return not (s.isalpha() or s.isdigit())


def apply_chunk_merging(tokens, next_tokens, punc_dict={':', ".", ",", "?"}, overlap_size=8, min_words_cut=4):
    num_words = 0
    num_words_cut = 0
    head_idx = tail_idx = 0

    # Return next tokens if current tokens list is empty
    if not tokens:
        return next_tokens
    
    for token in tokens[::-1]:
        if token not in punc_dict:
            if check_alnum(token):
                clean_token = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", token)
                clean_token = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", clean_token)
                word_len = len(clean_token.split())
            else:
                word_len = 1
            
            if num_words_cut + word_len > min_words_cut:
                break
            
            num_words_cut += word_len
        tail_idx += 1
    
    tokens = tokens[:-tail_idx]
    
    num_words_pass = overlap_size - num_words_cut
    for token in next_tokens:
        if token not in punc_dict:
            sub_tokens = []
            if check_alnum(token):
                clean_token = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", token)
                clean_token = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", clean_token)
                sub_tokens = clean_token.split()
                num_words += len(sub_tokens)
            else:
                num_words += 1
            
            if num_words >= num_words_pass:
                head_idx += 1
                if num_words > num_words_pass:
                    idx = num_words - num_words_pass
                    tokens.append("".join(sub_tokens[len(sub_tokens) - idx:]))
                break

        head_idx += 1

    tokens.extend(next_tokens[head_idx:])
    return tokens


def merge_chunk(batch, overlap_size=8, min_words_cut=4):
    result = []
    punc_dict = {':', ".", ",", "?"}
    if len(batch) == 1 or overlap_size == 0:
        for sub_tokens in batch:
            result.extend(sub_tokens)
    else:
        for _, sub_tokens in enumerate(batch):
            try:
                result = apply_chunk_merging(result, sub_tokens, punc_dict, overlap_size, min_words_cut)
            except Exception as e:
                print(e)

    result = " ".join(result)
    # result = re.sub(r'(\s[\.\,\?\:]){2,}', r'\1', result)
    return result


def main(args):
    if args.split_chunk and args.overlap_size < args.min_words_cut:
        raise ValueError("Min words cut must be smaller than or equal to overlap size when using chunk merging")
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size, split_chunk=args.split_chunk,
                                       chunk_size=args.chunk_size, overlap_size=args.overlap_size,
                                       min_words_cut=args.min_words_cut)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=64)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        action='store_true',
                        help='Whether to lowercase tokens.',)
    parser.add_argument('--transformer_model',
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        action='store_true',
                        help='Whether to do ensembling.',)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--split_chunk',
                        action='store_true',
                        help='Whether to use chunk merging or not')
    parser.add_argument('--chunk_size',
                        type=int,
                        help='Chunk size for chunk merging',
                        default=32)
    parser.add_argument('--overlap_size',
                        type=int,
                        help='Overlapped words between two continuous chunks',
                        default=8)
    parser.add_argument('--min_words_cut',
                        type=int,
                        help='number of words at the end the first chunk to be removed during merge',
                        default=4)
    args = parser.parse_args()
    main(args)
