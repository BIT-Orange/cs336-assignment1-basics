from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

from cs336_basics.bpe_train import GPT2_PRETOKENIZER_PATTERN


Pair = tuple[bytes, bytes]


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    unicode_values = byte_values[:]
    offset = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


def _decode_gpt2_token(token: str, decoder: dict[str, int]) -> bytes:
    return bytes(decoder[char] for char in token)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.id_by_token = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {merge: rank for rank, merge in enumerate(self.merges)}
        self.pretokenizer_pattern = re.compile(GPT2_PRETOKENIZER_PATTERN)

        self.special_tokens = special_tokens or []
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.id_by_token:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.id_by_token[token_bytes] = token_id

        self.special_token_bytes = {token.encode("utf-8"): token for token in self.special_tokens}
        self.special_pattern = self._compile_special_pattern(self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab_data = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))

        if all(isinstance(token_id, str) and token_id.isdigit() for token_id in vocab_data):
            vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in vocab_data.items()}
        else:
            gpt2_decoder = {char: byte for byte, char in _gpt2_bytes_to_unicode().items()}
            vocab = {token_id: _decode_gpt2_token(token, gpt2_decoder) for token, token_id in vocab_data.items()}

        merges: list[Pair] = []
        gpt2_decoder = {char: byte for byte, char in _gpt2_bytes_to_unicode().items()}
        for line in Path(merges_filepath).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            left, right = line.split()
            try:
                merges.append((bytes.fromhex(left), bytes.fromhex(right)))
            except ValueError:
                merges.append((_decode_gpt2_token(left, gpt2_decoder), _decode_gpt2_token(right, gpt2_decoder)))

        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _compile_special_pattern(special_tokens: list[str]) -> re.Pattern[str] | None:
        if not special_tokens:
            return None
        escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        return re.compile("|".join(escaped_tokens))

    def _encode_pretoken_bytes(self, pretoken: bytes) -> list[int]:
        parts = [bytes([byte]) for byte in pretoken]

        while len(parts) > 1:
            candidate: tuple[int, Pair] | None = None
            for i, pair in enumerate(zip(parts, parts[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (candidate is None or rank < candidate[0]):
                    candidate = (rank, pair)

            if candidate is None:
                break

            _, pair_to_merge = candidate
            merged = pair_to_merge[0] + pair_to_merge[1]
            next_parts: list[bytes] = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == pair_to_merge[0] and parts[i + 1] == pair_to_merge[1]:
                    next_parts.append(merged)
                    i += 2
                else:
                    next_parts.append(parts[i])
                    i += 1
            parts = next_parts

        return [self.id_by_token[token] for token in parts]

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        for match in self.pretokenizer_pattern.finditer(text):
            yield from self._encode_pretoken_bytes(match.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(self._encode_text(text))

    def _encode_text(self, text: str) -> Iterator[int]:
        if self.special_pattern is None:
            yield from self._encode_ordinary_text(text)
            return

        start = 0
        for match in self.special_pattern.finditer(text):
            yield from self._encode_ordinary_text(text[start : match.start()])
            yield self.id_by_token[match.group().encode("utf-8")]
            start = match.end()
        yield from self._encode_ordinary_text(text[start:])

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.special_tokens:
            max_special_len = max(len(token) for token in self.special_tokens)
        else:
            max_special_len = 0

        carry = ""
        for chunk in iterable:
            text = carry + chunk
            if not text:
                continue

            safe_end = self._safe_encode_prefix_end(text, max_special_len)
            if safe_end == 0:
                carry = text
                continue

            yield from self._encode_text(text[:safe_end])
            carry = text[safe_end:]

        if carry:
            yield from self._encode_text(carry)

    def _safe_encode_prefix_end(self, text: str, max_special_len: int) -> int:
        keep_chars = max(max_special_len - 1, 0)
        candidate_end = max(0, len(text) - keep_chars)
        if candidate_end <= 0:
            return 0

        if self.special_pattern is not None:
            for match in self.special_pattern.finditer(text):
                if match.start() < candidate_end < match.end():
                    candidate_end = match.start()
                    break

        last_match_end = 0
        for match in self.pretokenizer_pattern.finditer(text):
            if match.end() <= candidate_end and match.end() < len(text):
                last_match_end = match.end()
            elif match.start() >= candidate_end:
                break
        return last_match_end

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
