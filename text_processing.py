import re
import inflect
import unicodedata
import torch
import numpy as np


class CTCTokenizer:
    def __init__(self):
        # Define character set (adjust as needed for your language)
        chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                 ' ']

        # Add special tokens
        self.blank_token = '<blank>'  # Blank token for CTC
        self.unk_token = '<unk>'  # Unknown character token

        # Create vocabulary with blank token at index 0 (common for CTC)
        self.vocab = [self.blank_token] + chars + [self.unk_token]

        # Create mappings
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

        # Vocabulary size
        self.vocab_size = len(self.vocab)

        self.p = inflect.engine()

    def remove_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if not unicodedata.combining(c))

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace

        return text

    def numbers_to_words(self, text):

        def convert_number(match):
            number_str = match.group()
            if '.' in number_str:
                # Split on dot, convert each part separately
                left, right = number_str.split('.')
                left_word = self.p.number_to_words(left).replace("-", " ")
                right_word = self.p.number_to_words(right).replace("-", " ")
                return f"{left_word} {right_word}"
            else:
                word = self.p.number_to_words(number_str)
                return word.replace("-", " ")

        # Match integers and decimals
        return re.sub(r'\b\d+(\.\d+)?\b', convert_number, text)

    def remove_oov_characters(self, text):
        return ''.join(c for c in text if c in self.vocab)

    def encode(self, text, max_length=128):

        text = self.remove_accents(text)
        text = self.numbers_to_words(text)
        text = self.normalize_text(text)
        text = self.remove_oov_characters(text)
        #print("preprocessed text: ", text)

        # Convert characters to indices
        indices = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in text]

        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif len(indices) < max_length:
                indices = indices + [self.char2idx[self.blank_token]] * (max_length - len(indices))

        return np.array(indices, dtype=np.int32)

    def load_alignment(self, path, max_length=128):
        with open(path, 'r') as f:
            lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.strip().split()
            if line[2] != 'sil':
                tokens.append(line[2])
                tokens.append(' ')

        merged = ''.join(tokens)
        indices = [self.char2idx.get(c, self.char2idx[self.unk_token]) for c in merged.strip()]
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif len(indices) < max_length:
                indices = indices + [self.char2idx[self.blank_token]] * (max_length - len(indices))
        return indices

    # def _decode_greedy(self, indices):
    #
    #     # if isinstance(indices, torch.Tensor):
    #     #    batch_indices = batch_indices.cpu().numpy()
    #     if indices.dim() == 2:
    #         indices = torch.argmax(indices, dim=-1)  # [num_seq,]
    #     indices = torch.unique_consecutive(indices, dim=-1)
    #     indices = indices.cpu().tolist()
    #     #indices = [i for i in indices if i != self.char2idx[self.blank_token]]
    #     #joined = "".join([self.idx2char[i] for i in indices])
    #     #final = joined.replace("|", " ").strip().split()
    #     #return joined
    #     decoded = []
    #     for i in indices:
    #         if i == self.char2idx[self.blank_token] or i == self.char2idx[self.unk_token]:
    #             continue
    #         elif i >= len(self.vocab):  # catch OOV errors
    #             continue
    #         decoded.append(self.idx2char[i])
    #
    #     return "".join(decoded)

    def _decode_greedy(self, input_seq):
        """
        input_seq: torch.Tensor or np.ndarray of shape (..., vocab_size) or (...,)
        Returns:
            str: Decoded string
        """
        # Convert to numpy if tensor
        if isinstance(input_seq, torch.Tensor):
            if input_seq.dim() == 2:  # assume shape (T, vocab_size)
                input_seq = torch.argmax(input_seq, dim=-1)
            input_seq = input_seq.cpu().numpy()

        elif isinstance(input_seq, np.ndarray) and input_seq.ndim == 2:
            input_seq = np.argmax(input_seq, axis=-1)

        # input_seq is now shape (T,)
        input_seq = np.array(input_seq)
        input_seq = np.array([input_seq[0]] + [i for i, j in zip(input_seq[1:], input_seq[:-1]) if i != j])

        decoded = []
        for i in input_seq:
            if i == self.char2idx.get(self.blank_token) or i == self.char2idx.get(self.unk_token):
                continue
            if i >= len(self.vocab):  # OOV protection
                continue
            decoded.append(self.idx2char[i])

        return "".join(decoded)

    def decode_greedy_batch(self, batch_indices):

        batch_results = []
        for indices in batch_indices:
            result = self._decode_greedy(indices)
            batch_results.append(result)

        return batch_results

    def decode_beam_search(self, logits, lengths=None, beam_width=100, top_paths=1):

        batch_size = logits.shape[0]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().detach().numpy()
        results = []

        # Process each item in the batch
        for i in range(batch_size):
            # Apply sequence mask if lengths provided
            if lengths is not None:
                seq_len = lengths[i].item()
                sequence = log_probs[i, :seq_len]
            else:
                sequence = log_probs[i]

            # Run beam search on this sequence
            paths = self._beam_search_single(sequence, beam_width, top_paths)

            if top_paths == 1:
                # Return only the best path's text
                results.append(paths[0][0])
            else:
                # Return top N paths with their scores
                results.append(paths)

        return results

    def _beam_search_single(self,log_probs, beam_width, top_paths):

        # Blank and non-blank prefixes with their probabilities
        T = log_probs.shape[0]  # Sequence length

        # Initialize with empty prefix (using a special token)
        # Each prefix is represented as (prefix_str, (p_blank, p_non_blank))
        # where p_blank is probability ending in blank, p_non_blank is probability ending in non-blank
        prefixes = {(): (0.0, float('-inf'))}  # Empty prefix starts with p_blank=1 (log=0)

        # Process each timestep
        for t in range(T):
            # Collect new prefixes
            new_prefixes = {}

            # For each existing prefix
            for prefix, (p_b, p_nb) in prefixes.items():
                # For each possible new character
                for c in range(len(self.idx2char)):
                    c_log_prob = log_probs[t, c]

                    # Case 1: Extending with blank (doesn't change prefix)
                    if c == self.blank_token:
                        # New p_blank for this prefix
                        new_p_b = np.logaddexp(p_b + c_log_prob, p_nb + c_log_prob)

                        # Add to new_p_blank or create new entry
                        if prefix in new_prefixes:
                            new_prefixes[prefix] = (
                            np.logaddexp(new_prefixes[prefix][0], new_p_b), new_prefixes[prefix][1])
                        else:
                            new_prefixes[prefix] = (new_p_b, float('-inf'))
                    else:
                        # Get character from label index
                        c_char = self.idx2char[c]

                        # Case 2: Adding new character (different from last one)
                        new_prefix = prefix + (c,)

                        # Case 2.1: New prefix is created by extending prefix with c
                        new_p_nb = p_nb + c_log_prob

                        # Add to new_p_non_blank or create new entry
                        if new_prefix in new_prefixes:
                            new_prefixes[new_prefix] = (
                            new_prefixes[new_prefix][0], np.logaddexp(new_prefixes[new_prefix][1], new_p_nb))
                        else:
                            new_prefixes[new_prefix] = (float('-inf'), new_p_nb)

                        # Case 2.2: Extension from blank
                        new_p_nb_from_blank = p_b + c_log_prob
                        if new_prefix in new_prefixes:
                            new_prefixes[new_prefix] = (
                            new_prefixes[new_prefix][0], np.logaddexp(new_prefixes[new_prefix][1], new_p_nb_from_blank))
                        else:
                            new_prefixes[new_prefix] = (float('-inf'), new_p_nb_from_blank)

                        # Case 3: Same character repeated
                        if prefix and prefix[-1] == c:
                            # We add log_prob to p_nb, but we don't create a new prefix
                            repeated_prefix = prefix
                            new_p_nb_repeat = p_nb + c_log_prob

                            if repeated_prefix in new_prefixes:
                                new_prefixes[repeated_prefix] = (new_prefixes[repeated_prefix][0],
                                                                 np.logaddexp(new_prefixes[repeated_prefix][1],
                                                                              new_p_nb_repeat))
                            else:
                                new_prefixes[repeated_prefix] = (float('-inf'), new_p_nb_repeat)

            # Sort by total log probability and keep top beam_width
            prefixes = {}
            prefix_list = [(p, np.logaddexp(pb, pnb)) for p, (pb, pnb) in new_prefixes.items()]
            prefix_list.sort(key=lambda x: x[1], reverse=True)

            for p, score in prefix_list[:beam_width]:
                prefixes[p] = new_prefixes[p]

        # Get top paths
        final_paths = [(p, np.logaddexp(pb, pnb)) for p, (pb, pnb) in prefixes.items()]
        final_paths.sort(key=lambda x: x[1], reverse=True)

        # Convert indices to strings
        string_paths = []
        for prefix, score in final_paths[:top_paths]:
            text = ''.join(self.idx2char[c] for c in prefix)
            string_paths.append((text, score))

        return string_paths

    def decode_prefix_beam_search(self, logits, lengths=None, beam_width=100, top_paths=1):

        return self.decode_beam_search(logits, lengths, beam_width, top_paths)

    def get_vocab_size(self):
        return self.vocab_size