import re, os
import html
import unicodedata
import numpy as np
import torch
import inflect


class TextTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyz ")

        self.blank_token = "<blank>"
        self.blank_id = 0

        self.vocab = [self.blank_token] + chars

        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

        self.vocab_size = len(self.vocab)

        self.p = inflect.engine()

    def remove_accents(self, text):
        return "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )

    def numbers_to_words(self, text):
        def convert_number(match):
            number_str = match.group()

            if "." in number_str:
                left, right = number_str.split(".")
                left_word = self.p.number_to_words(left)
                right_word = self.p.number_to_words(right)
                word = f"{left_word} {right_word}"
            else:
                word = self.p.number_to_words(number_str)

            word = word.replace("-", " ")
            word = word.replace(",", " ")
            return word

        return re.sub(r"\b\d+(\.\d+)?\b", convert_number, text)

    def normalize_text(self, text):
        text = html.unescape(text)
        text = text.lower()
        text = self.remove_accents(text)

        # Convert numbers before deleting non-letter symbols
        text = self.numbers_to_words(text)

        # Optional: normalize common apostrophe styles before removing them
        text = text.replace("’", "'")
        text = text.replace("`", "'")

        # Remove apostrophes by joining contractions:
        # don't -> dont, i'm -> im
        text = text.replace("'", "")

        # Keep only lowercase English letters and spaces
        text = re.sub(r"[^a-z\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def encode(self, text, max_length=None):
        text = self.normalize_text(text)

        indices = [self.char2idx[c] for c in text if c in self.char2idx]
        length = len(indices)

        if max_length is not None:
            if length > max_length:
                indices = indices[:max_length]
                length = max_length
            else:
                indices = indices + [self.blank_id] * (max_length - length)

        return np.array(indices, dtype=np.int32), length

    def decode(self, encoded, length=None):
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()

        encoded = np.array(encoded).flatten()

        if length is not None:
            encoded = encoded[:length]

        chars = []

        for idx in encoded:
            idx = int(idx)

            if idx == self.blank_id:
                continue

            if idx in self.idx2char:
                chars.append(self.idx2char[idx])

        return "".join(chars)

    def decode_index_batch(self, indices_batch) -> list[str]:
        """
        Decode a batch of target/index sequences or logits.

        Accepts:
            [B, T]    indices
            [B, T, V] logits/probs

        """
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.detach().cpu().numpy()

        indices_batch = np.asarray(indices_batch)

        if indices_batch.ndim == 3:
            indices_batch = np.argmax(indices_batch, axis=-1)

        results = []

        for indices in indices_batch:
            results.append(self.decode(indices))

        return results

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
        indices = [self.char2idx[c] for c in merged.strip() if c in self.char2idx]
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif len(indices) < max_length:
                indices = indices + [self.char2idx[self.blank_token]] * (max_length - len(indices))
        return indices

    def read_transcripts_for_chunk(self,
            txt_path: str,
            start_sec: float = 0.0,
            max_sec: float = 3.0,
            require_full_word_inside: bool = True,
            min_overlap_ratio: float = 0.5,
    ) -> tuple[str, list[tuple[str, float, float]]]:
        """
        Read WhisperX-style timed transcript.

        Accepts lines like:
            WORD START END
            WORD START END SCORE

        Keeps words that belong to [start_sec, start_sec + max_sec].

        Returns:
            transcript: selected words joined by space
            timed_words: [(word, start, end), ...]
        """
        end_sec = start_sec + max_sec

        if not os.path.isfile(txt_path):
            return "", []

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]

        selected = []

        for line in lines:
            if not line or line.startswith("Text:"):
                continue

            parts = line.split()
            if len(parts) not in (3, 4):
                continue

            word = parts[0]

            try:
                w_start = float(parts[1])
                w_end = float(parts[2])
            except ValueError:
                continue

            if w_end <= w_start:
                continue

            if require_full_word_inside:
                keep = (w_start >= start_sec) and (w_end <= end_sec)
            else:
                overlap = max(0.0, min(w_end, end_sec) - max(w_start, start_sec))
                duration = max(1e-6, w_end - w_start)
                keep = (overlap / duration) >= min_overlap_ratio

            if keep:
                selected.append((word, w_start, w_end))

        words = [w for w, _, _ in selected]
        transcript = " ".join(words)
        transcript = re.sub(r"\s+", " ", transcript).strip()

        return transcript, selected

    def decode_greedy(self, input_seq):
        """
        CTC greedy decoding for a single sequence.

        input_seq:
            [T, V] logits/probabilities/log-probabilities
            or
            [T] predicted class IDs

        Returns:
            decoded string
        """

        # Convert logits/probs to class IDs
        if isinstance(input_seq, torch.Tensor):
            if input_seq.dim() == 2:  # [T, V]
                input_seq = torch.argmax(input_seq, dim=-1)

            input_seq = input_seq.detach().cpu().numpy()

        elif isinstance(input_seq, np.ndarray):
            if input_seq.ndim == 2:  # [T, V]
                input_seq = np.argmax(input_seq, axis=-1)

        input_seq = np.array(input_seq).flatten()

        if len(input_seq) == 0:
            return ""

        blank_idx = self.char2idx[self.blank_token]
        unk_idx = None
        if hasattr(self, "unk_token"):
            unk_idx = self.char2idx.get(self.unk_token, None)

        decoded = []
        prev_idx = None

        for idx in input_seq:
            idx = int(idx)

            # CTC collapse repeated symbols
            if idx == prev_idx:
                continue

            prev_idx = idx

            # Remove CTC blank
            if idx == blank_idx:
                continue

            # Optionally remove unknown token
            if unk_idx is not None and idx == unk_idx:
                continue

            # OOV protection
            if idx not in self.idx2char:
                continue

            decoded.append(self.idx2char[idx])

        return "".join(decoded)

    def decode_greedy_batch(self, batch_indices):
        """
        batch_indices:
            [B, T, V] logits/probs/log-probs
            or
            [B, T] predicted class IDs

        Returns:
            list[str]
        """

        batch_results = []

        for indices in batch_indices:
            result = self.decode_greedy(indices)
            batch_results.append(result)

        return batch_results

    def _to_log_probs(self, x):
        """
        Convert logits/probs/log_probs of shape [T, V] to log_probs.

        Assumes raw logits by default and applies stable log-softmax.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError(
                f"decode_beam_search expects shape [T, V], but got {x.shape}"
            )

        # Stable log-softmax
        x_max = np.max(x, axis=-1, keepdims=True)
        log_probs = x - x_max - np.log(
            np.sum(np.exp(x - x_max), axis=-1, keepdims=True)
        )

        return log_probs

    def decode_beam_search(
            self,
            logits,
            beam_width: int = 20,
            top_paths: int = 1,
    ):
        """
        CTC prefix beam search decoding for one sequence.

        Args:
            logits:
                Tensor/array of shape [T, V].
                Usually raw model logits before softmax.
            beam_width:
                Number of beams to keep. Default: 20.
            top_paths:
                Number of decoded hypotheses to return.

        Returns:
            If top_paths == 1:
                str
            Else:
                list of (decoded_text, score)
        """

        log_probs = self._to_log_probs(logits)

        blank_id = self.blank_id
        vocab_size = self.vocab_size

        # prefix -> (prob ending with blank, prob ending with non-blank)
        # probabilities are stored in log-space
        beams = {
            (): (0.0, float("-inf"))
        }

        T, V = log_probs.shape

        if V != vocab_size:
            raise ValueError(
                f"Expected vocab size {vocab_size}, but logits have size {V}"
            )

        for t in range(T):
            new_beams = {}

            for prefix, (p_blank, p_nonblank) in beams.items():
                for c in range(V):
                    p = float(log_probs[t, c])

                    # Case 1: CTC blank
                    if c == blank_id:
                        new_p_blank = np.logaddexp(
                            p_blank + p,
                            p_nonblank + p
                        )

                        old_p_blank, old_p_nonblank = new_beams.get(
                            prefix,
                            (float("-inf"), float("-inf"))
                        )

                        new_beams[prefix] = (
                            np.logaddexp(old_p_blank, new_p_blank),
                            old_p_nonblank,
                        )

                    # Case 2: non-blank character
                    else:
                        last_char = prefix[-1] if len(prefix) > 0 else None

                        # Repeating same character
                        if c == last_char:
                            # Same char repeated without blank:
                            # stay at same prefix
                            old_p_blank, old_p_nonblank = new_beams.get(
                                prefix,
                                (float("-inf"), float("-inf"))
                            )

                            new_beams[prefix] = (
                                old_p_blank,
                                np.logaddexp(
                                    old_p_nonblank,
                                    p_nonblank + p
                                ),
                            )

                            # Same char repeated after blank:
                            # extend prefix
                            new_prefix = prefix + (c,)

                            old_p_blank, old_p_nonblank = new_beams.get(
                                new_prefix,
                                (float("-inf"), float("-inf"))
                            )

                            new_beams[new_prefix] = (
                                old_p_blank,
                                np.logaddexp(
                                    old_p_nonblank,
                                    p_blank + p
                                ),
                            )

                        # New different character
                        else:
                            new_prefix = prefix + (c,)

                            new_p_nonblank = np.logaddexp(
                                p_blank + p,
                                p_nonblank + p,
                            )

                            old_p_blank, old_p_nonblank = new_beams.get(
                                new_prefix,
                                (float("-inf"), float("-inf"))
                            )

                            new_beams[new_prefix] = (
                                old_p_blank,
                                np.logaddexp(
                                    old_p_nonblank,
                                    new_p_nonblank
                                ),
                            )

            # Keep only top beam_width beams
            beams = dict(
                sorted(
                    new_beams.items(),
                    key=lambda item: np.logaddexp(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        # Final ranking
        final_beams = []

        for prefix, (p_blank, p_nonblank) in beams.items():
            score = np.logaddexp(p_blank, p_nonblank)

            text = "".join(
                self.idx2char[idx]
                for idx in prefix
                if idx != blank_id and idx in self.idx2char
            )

            final_beams.append((text, score))

        final_beams.sort(key=lambda x: x[1], reverse=True)

        if top_paths == 1:
            return final_beams[0][0]

        return final_beams[:top_paths]

    def decode_beam_search_batch(
            self,
            batch_logits,
            lengths=None,
            beam_width: int = 20,
            top_paths: int = 1,
    ):
        """
        CTC prefix beam search decoding for a batch.

        Args:
            batch_logits:
                Tensor/array of shape [B, T, V].
            lengths:
                Optional valid lengths for each sample.
            beam_width:
                Beam width. Default: 20.
            top_paths:
                Number of hypotheses per sample.

        Returns:
            If top_paths == 1:
                list[str]
            Else:
                list[list[(text, score)]]
        """

        if isinstance(batch_logits, torch.Tensor):
            batch_logits = batch_logits.detach().cpu().numpy()

        batch_logits = np.asarray(batch_logits)

        if batch_logits.ndim != 3:
            raise ValueError(
                f"decode_beam_search_batch expects shape [B, T, V], "
                f"but got {batch_logits.shape}"
            )

        results = []

        for b in range(batch_logits.shape[0]):
            logits = batch_logits[b]

            if lengths is not None:
                valid_len = int(lengths[b])
                logits = logits[:valid_len]

            decoded = self.decode_beam_search(
                logits,
                beam_width=beam_width,
                top_paths=top_paths,
            )

            results.append(decoded)

        return results

    def get_vocab_size(self):
        return self.vocab_size

class PhonemeTokenizer:
    """
    English phoneme tokenizer for CTC using g2p-en + ARPAbet.

    Vocabulary:
        0     -> <blank>  CTC blank
        1..39 -> ARPAbet phones without stress

    No <unk> token is used as a CTC target class.
    Unexpected g2p tokens are logged and skipped.
    """

    _ARPABET_PHONES = [
        # vowels
        "AA", "AE", "AH", "AO", "AW", "AY",
        "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",

        # consonants
        "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K",
        "L", "M", "N", "NG", "P", "R", "S", "SH", "T",
        "TH", "V", "W", "Y", "Z", "ZH",
    ]

    def __init__(self, log_unknown_tokens: bool = True):
        from g2p_en import G2p

        self.blank_token = "<blank>"
        self.blank_idx = 0

        self.vocab = [self.blank_token] + self._ARPABET_PHONES
        self.phone2idx = {p: i for i, p in enumerate(self.vocab)}
        self.idx2phone = {i: p for i, p in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.g2p = G2p()
        self.log_unknown_tokens = log_unknown_tokens
        #self.logger = logging.getLogger(self.__class__.__name__)

    def _remove_accents(self, text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )

    def _clean_transcript_text(self, text: str) -> str:
        text = self._remove_accents(text)
        text = text.lower()

        # Remove common non-speech annotations.
        text = re.sub(r"\[[^\]]*\]", " ", text)   # [noise], [laugh]
        text = re.sub(r"<[^>]*>", " ", text)      # <unk>, <laugh>

        # Keep English letters, digits, apostrophes, and whitespace.
        # g2p-en can handle numbers/contractions reasonably.
        text = re.sub(r"[^a-z0-9'\s]", " ", text)

        # Remove isolated apostrophes.
        text = re.sub(r"\s+'\s+", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def text_to_phones(self, text: str) -> list[str]:
        """
        Convert text to ARPAbet phones without stress digits.

        Example:
            "I need you" -> ["AY", "N", "IY", "D", "Y", "UW"]
        """
        original_text = text
        text = self._clean_transcript_text(text)

        if not text:
            return []

        raw = self.g2p(text)

        phones = []
        unknown_tokens = []

        for token in raw:
            token = str(token).strip()

            if not token:
                continue

            # Remove lexical stress.
            # Examples: AH0 -> AH, IY1 -> IY, ER2 -> ER
            phone = re.sub(r"[0-2]$", "", token)

            if phone in self.phone2idx and phone != self.blank_token:
                phones.append(phone)
            else:
                # g2p-en often returns spaces/punctuation-like tokens.
                # Log only tokens that look nontrivial.
                if not re.fullmatch(r"[\s.,!?;:'\"-]+", token):
                    unknown_tokens.append(token)

        if self.log_unknown_tokens and unknown_tokens:
            print(
                "Skipped unknown g2p token(s) %s from cleaned text=%r original text=%r",
                unknown_tokens,
                text,
                original_text,
            )

        return phones

    def encode(
        self,
        text: str,
        max_length: int | None = 128
    ) -> tuple[np.ndarray, int, list[str]]:
        """
        Return:
            indices: padded phone indices
            length: true CTC target length before padding
            phones: phone string list for debugging

        Note:
            Padding uses blank_idx only for batching.
            You must pass `length` as target_lengths to CTCLoss.
        """
        phones = self.text_to_phones(text)
        indices = [self.phone2idx[p] for p in phones]

        true_length = len(indices)

        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
                phones = phones[:max_length]
                true_length = max_length
            else:
                pad_len = max_length - len(indices)
                indices = indices + [self.blank_idx] * pad_len

        return np.array(indices, dtype=np.int32), int(true_length), phones

    def decode(self, indices, separator: str = " ") -> str:
        """
        Decode ground-truth phoneme indices to a string.

        This is for encoded targets.
        It removes blank padding but does NOT collapse repeated phones.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()

        indices = np.asarray(indices).flatten().astype(int)

        phones = []

        for idx in indices:
            idx = int(idx)

            if idx == self.blank_idx:
                continue

            if idx not in self.idx2phone:
                continue

            phones.append(self.idx2phone[idx])

        return separator.join(phones)

    def decode_index_batch(self, indices_batch) -> list[str]:
        """
        Decode a batch of target/index sequences or logits.

        Accepts:
            [B, T]    indices
            [B, T, V] logits/probs

        For model predictions, prefer decode_greedy_batch().
        """
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.detach().cpu().numpy()

        indices_batch = np.asarray(indices_batch)

        if indices_batch.ndim == 3:
            indices_batch = np.argmax(indices_batch, axis=-1)

        results = []

        for indices in indices_batch:
            results.append(self.decode(indices))

        return results

    def decode_greedy(self, input_seq, separator: str = " ") -> str:
        """
        CTC greedy decoding for one sample.

        Accepts:
            [T, V] logits/probs/log-probs
            [T]    predicted class ids

        Applies:
            1. argmax if input is logits/probs
            2. collapse repeated tokens
            3. remove CTC blank
            4. map indices to phonemes
        """
        if isinstance(input_seq, torch.Tensor):
            input_seq = input_seq.detach().cpu().numpy()

        input_seq = np.asarray(input_seq)

        if input_seq.ndim == 2:
            input_seq = np.argmax(input_seq, axis=-1)

        input_seq = input_seq.flatten().astype(int)

        phones = []
        prev_idx = None

        for idx in input_seq:
            idx = int(idx)

            # CTC collapse repeated tokens.
            if idx == prev_idx:
                continue

            prev_idx = idx

            # Remove CTC blank.
            if idx == self.blank_idx:
                continue

            # Skip invalid indices.
            if idx not in self.idx2phone:
                if self.log_unknown_tokens:
                    print("Skipped invalid predicted phone index: %s", idx)
                continue

            phones.append(self.idx2phone[idx])

        return separator.join(phones)

    def decode_greedy_batch(self, batch_input, lengths=None) -> list[str]:
        """
        CTC greedy decoding for a batch.

        Accepts:
            [B, T, V] logits/probs/log-probs
            [B, T]    predicted class ids
        """
        if isinstance(batch_input, torch.Tensor):
            batch_input = batch_input.detach().cpu().numpy()

        batch_input = np.asarray(batch_input)

        results = []

        for b in range(batch_input.shape[0]):
            seq = batch_input[b]

            if lengths is not None:
                seq = seq[:int(lengths[b])]

            results.append(self.decode_greedy(seq))

        return results

    def _logsumexp2(self, a, b):
        """
        Stable log(exp(a) + exp(b)) for two log-probabilities.
        """
        if a == float("-inf"):
            return b

        if b == float("-inf"):
            return a

        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    def _to_log_probs(self, x):
        """
        Convert logits/probs/log_probs of shape [T, V] to log_probs.

        Assumes raw logits by default.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError(
                f"Beam search expects logits/probs/log_probs with shape [T, V], got shape {x.shape}"
            )

        # Treat input as raw logits and apply stable log-softmax.
        x_max = np.max(x, axis=-1, keepdims=True)
        log_probs = x - x_max - np.log(
            np.sum(np.exp(x - x_max), axis=-1, keepdims=True)
        )

        return log_probs

    def decode_beam_search(
        self,
        logits,
        beam_width: int = 20,
        top_paths: int = 1,
        separator: str = " ",
    ):
        """
        CTC prefix beam search decoding for one sample.

        Args:
            logits: shape [T, V], raw model logits
            beam_width: number of active beams to keep
            top_paths: number of decoded hypotheses to return
            separator: separator between phonemes

        Returns:
            str if top_paths == 1
            else list[(decoded_string, score)]
        """
        log_probs = self._to_log_probs(logits)

        blank_idx = self.blank_idx

        beams = {
            (): (0.0, float("-inf"))
        }

        T, V = log_probs.shape

        for t in range(T):
            new_beams = {}

            for prefix, (p_blank, p_nonblank) in beams.items():
                for c in range(V):
                    p = float(log_probs[t, c])

                    # Invalid class index.
                    if c not in self.idx2phone:
                        continue

                    # Case 1: emit blank.
                    if c == blank_idx:
                        new_p_blank = self._logsumexp2(
                            p_blank + p,
                            p_nonblank + p,
                        )

                        old_p_blank, old_p_nonblank = new_beams.get(
                            prefix,
                            (float("-inf"), float("-inf")),
                        )

                        new_beams[prefix] = (
                            self._logsumexp2(old_p_blank, new_p_blank),
                            old_p_nonblank,
                        )

                    # Case 2: emit phone.
                    else:
                        last = prefix[-1] if len(prefix) > 0 else None

                        if c == last:
                            # Repeated phone without blank: stay at same prefix.
                            old_p_blank, old_p_nonblank = new_beams.get(
                                prefix,
                                (float("-inf"), float("-inf")),
                            )

                            new_beams[prefix] = (
                                old_p_blank,
                                self._logsumexp2(
                                    old_p_nonblank,
                                    p_nonblank + p,
                                ),
                            )

                            # Repeated phone after blank: extend prefix.
                            new_prefix = prefix + (c,)

                            old_p_blank, old_p_nonblank = new_beams.get(
                                new_prefix,
                                (float("-inf"), float("-inf")),
                            )

                            new_beams[new_prefix] = (
                                old_p_blank,
                                self._logsumexp2(
                                    old_p_nonblank,
                                    p_blank + p,
                                ),
                            )

                        else:
                            # Different phone: extend prefix normally.
                            new_prefix = prefix + (c,)

                            new_p_nonblank = self._logsumexp2(
                                p_blank + p,
                                p_nonblank + p,
                            )

                            old_p_blank, old_p_nonblank = new_beams.get(
                                new_prefix,
                                (float("-inf"), float("-inf")),
                            )

                            new_beams[new_prefix] = (
                                old_p_blank,
                                self._logsumexp2(
                                    old_p_nonblank,
                                    new_p_nonblank,
                                ),
                            )

            beams = dict(
                sorted(
                    new_beams.items(),
                    key=lambda item: self._logsumexp2(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        final_beams = []

        for prefix, (p_blank, p_nonblank) in beams.items():
            score = self._logsumexp2(p_blank, p_nonblank)

            phones = [
                self.idx2phone[idx]
                for idx in prefix
                if idx != blank_idx and idx in self.idx2phone
            ]

            decoded = separator.join(phones)
            final_beams.append((decoded, score))

        final_beams.sort(key=lambda x: x[1], reverse=True)

        if top_paths == 1:
            return final_beams[0][0]

        return final_beams[:top_paths]

    def decode_beam_search_batch(
        self,
        batch_logits,
        lengths=None,
        beam_width: int = 20,
        top_paths: int = 1,
        separator: str = " ",
    ):
        """
        CTC prefix beam search decoding for a batch.

        Args:
            batch_logits: shape [B, T, V]
            lengths: optional valid lengths for each sample
            beam_width: beam width
            top_paths: number of hypotheses per sample
            separator: separator between phonemes

        Returns:
            list[str] if top_paths == 1
            list[list[(str, score)]] if top_paths > 1
        """
        if isinstance(batch_logits, torch.Tensor):
            batch_logits = batch_logits.detach().cpu().numpy()

        batch_logits = np.asarray(batch_logits)

        if batch_logits.ndim != 3:
            raise ValueError(
                f"Expected batch logits with shape [B, T, V], got {batch_logits.shape}"
            )

        results = []

        for b in range(batch_logits.shape[0]):
            logits = batch_logits[b]

            if lengths is not None:
                logits = logits[:int(lengths[b])]

            decoded = self.decode_beam_search(
                logits,
                beam_width=beam_width,
                top_paths=top_paths,
                separator=separator,
            )

            results.append(decoded)

        return results

    def get_vocab_size(self) -> int:
        return self.vocab_size