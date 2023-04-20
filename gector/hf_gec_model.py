"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
from collections import defaultdict
import logging
from time import time

import torch
from vocabulary import Vocabulary
from transformers import AutoTokenizer
from modeling_seq2labels import Seq2LabelsModel
from utils.helpers import PAD, UNK, START_TOKEN, get_target_sent_by_edits, get_weights_name

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class GecBERTModel(object):
    def __init__(self, vocab_path=None, model_paths=None,
                 weights=None,
                 max_len=64,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=3,
                 model_name='roberta',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 resolve_cycles=False,
                 ):
        self.model_weights = list(map(float, weights)) if weights else [1] * len(model_paths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.incorr_index = self.vocab.get_token_index("INCORRECT", "d_tags")
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles
        # set training parameters and operations

        self.indexers = []
        self.models = []
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        for model_path in model_paths:
            model = Seq2LabelsModel.from_pretrained(model_path)
            config = model.config
            model_name = config.pretrained_name_or_path
            special_tokens_fix = config.special_tokens_fix
            self.indexers.append(self._get_indexer(model_name, special_tokens_fix))
            model.eval().to(self.device)
            self.models.append(model)

    def _get_indexer(self, weights_name, special_tokens_fix):
        tokenizer = AutoTokenizer.from_pretrained(
            weights_name, do_basic_tokenize=False, do_lower_case=self.lowercase_tokens, model_max_length=1024
        )
        # to adjust all tokenizers
        if hasattr(tokenizer, 'encoder'):
            tokenizer.vocab = tokenizer.encoder
        if hasattr(tokenizer, 'sp_model'):
            tokenizer.vocab = defaultdict(lambda: 1)
            for i in range(tokenizer.sp_model.get_piece_size()):
                tokenizer.vocab[tokenizer.sp_model.id_to_piece(i)] = i

        if special_tokens_fix:
            tokenizer.add_tokens([START_TOKEN])
            tokenizer.vocab[START_TOKEN] = len(tokenizer) - 1
        return tokenizer

    def predict(self, batches):
        t11 = time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch = batch.to(self.device)
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_error_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            token_batch = [[START_TOKEN] + sequence[:max_len] for sequence in token_batch]
            batch = indexer(
                token_batch, return_tensors="pt", padding=True,
                is_split_into_words=True, truncation=True,
                add_special_tokens=False
            )
            offset_batch = []
            for i in range(len(token_batch)):
                word_ids = batch.word_ids(batch_index=i)
                offsets = [0]
                for i in range(1, len(word_ids)):
                    if word_ids[i] != word_ids[i - 1]:
                        offsets.append(i)
                offset_batch.append(torch.LongTensor(offsets))

            batch["input_offsets"] = torch.nn.utils.rnn.pad_sequence(
                offset_batch, batch_first=True, padding_value=0).to(torch.long)

            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['logits'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            class_probabilities_labels = torch.softmax(output['logits'], dim=-1)
            all_class_probs += weight * class_probabilities_labels / sum(self.model_weights)
            class_probabilities_d = torch.softmax(output['detect_logits'], dim=-1)
            error_probs_d = class_probabilities_d[:, :, self.incorr_index]
            incorr_prob = torch.max(error_probs_d, dim=-1)[0]
            error_probs += weight * incorr_prob / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            length = min(len(tokens), self.max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i],
                                                             namespace='labels')
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch):
        """
        Handle batch of requests.
        """
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break

        return final_batch, total_updates
