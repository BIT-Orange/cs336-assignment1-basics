from collections import Counter
from array import array
from cs336_basics.CountBuckets import CountBuckets

Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]

UINT32_MAX = (1 << 32) - 1
INT64_MAX = (1 << 63) - 1
PAIR_KEY_SHIFT = 32

class ArrayBpeTrainer:
    def __init__(
        self,
        token_counts: Counter[Token],
        symbol_bytes: list[bytes],
        symbol_map: dict[bytes, int],
        symbol_array_type: str,
    ) -> None:
        self.symbol_bytes = symbol_bytes
        self.symbol_map = symbol_map
        self.symbol_max = 0xFFFF if symbol_array_type == "H" else UINT32_MAX

        self.node_symbol = array(symbol_array_type, [0])
        self.node_next = array("I", [0])
        self.node_edge = array("I", [0])
        self.token_heads = array("I")
        self.token_weights = array("q")
        self.token_marks = array("I")
        self._mark_stamp = 0

        self.pair_map: dict[int, int] = {}
        self.pair_left = array(symbol_array_type, [0])
        self.pair_right = array(symbol_array_type, [0])
        self.pair_count = array("q", [0])
        self.pair_head = array("I", [0])
        self.pair_prev = array("I", [0])
        self.pair_next = array("I", [0])
        self.free_pairs = array("I")

        self.edge_token = array("I", [0])
        self.edge_pair = array("I", [0])
        self.edge_prev = array("I", [0])
        self.edge_next = array("I", [0])
        self.free_edges = array("I")

        self.total_edge_weight = 0
        self._consume_token_counts(token_counts)
        self.buckets = CountBuckets(
            self.pair_left,
            self.pair_right,
            self.pair_prev,
            self.pair_next,
            self.symbol_bytes,
            self.total_edge_weight,
        )
        for pair_id in range(1, len(self.pair_count)):
            count = self.pair_count[pair_id]
            if count > 0:
                self.buckets.add(pair_id, count)

    @staticmethod
    def _pair_key(left: int, right: int) -> int:
        return (left << PAIR_KEY_SHIFT) | right

    def _get_symbol_id(self, value: bytes) -> int:
        symbol_id = self.symbol_map.get(value)
        if symbol_id is not None:
            return symbol_id
        symbol_id = len(self.symbol_bytes)
        if symbol_id > self.symbol_max:
            raise OverflowError("too many BPE symbols for the compact symbol representation")
        self.symbol_map[value] = symbol_id
        self.symbol_bytes.append(value)
        return symbol_id

    def _get_pair_id(self, left: int, right: int) -> int:
        key = self._pair_key(left, right)
        pair_id = self.pair_map.get(key)
        if pair_id is not None:
            return pair_id

        if self.free_pairs:
            pair_id = self.free_pairs.pop()
            self.pair_left[pair_id] = left
            self.pair_right[pair_id] = right
            self.pair_count[pair_id] = 0
            self.pair_head[pair_id] = 0
            self.pair_prev[pair_id] = 0
            self.pair_next[pair_id] = 0
        else:
            pair_id = len(self.pair_count)
            if pair_id > INT64_MAX:
                raise OverflowError("too many BPE pairs for the compact int64 representation")
            self.pair_left.append(left)
            self.pair_right.append(right)
            self.pair_count.append(0)
            self.pair_head.append(0)
            self.pair_prev.append(0)
            self.pair_next.append(0)
        self.pair_map[key] = pair_id
        return pair_id

    def _append_edge(self, pair_id: int, left_node: int, token_id: int) -> None:
        if self.free_edges:
            edge_id = self.free_edges.pop()
            self.edge_token[edge_id] = token_id
            self.edge_pair[edge_id] = pair_id
            self.edge_prev[edge_id] = 0
            self.edge_next[edge_id] = self.pair_head[pair_id]
        else:
            edge_id = len(self.edge_token)
            if edge_id > INT64_MAX:
                raise OverflowError("too many BPE edges for the compact int64 representation")
            self.edge_token.append(token_id)
            self.edge_pair.append(pair_id)
            self.edge_prev.append(0)
            self.edge_next.append(self.pair_head[pair_id])

        old_head = self.pair_head[pair_id]
        if old_head:
            self.edge_prev[old_head] = edge_id
        self.pair_head[pair_id] = edge_id
        self.node_edge[left_node] = edge_id

    def _remove_edge(self, edge_id: int) -> int:
        pair_id = self.edge_pair[edge_id]
        previous = self.edge_prev[edge_id]
        following = self.edge_next[edge_id]
        if previous:
            self.edge_next[previous] = following
        else:
            self.pair_head[pair_id] = following
        if following:
            self.edge_prev[following] = previous

        self.edge_token[edge_id] = 0
        self.edge_pair[edge_id] = 0
        self.edge_prev[edge_id] = 0
        self.edge_next[edge_id] = 0
        self.free_edges.append(edge_id)
        return pair_id

    def _consume_token_counts(self, token_counts: Counter[Token]) -> None:
        # popitem avoids making a second list of all unique pre-token keys.
        while token_counts:
            token, raw_count = token_counts.popitem()
            count = int(raw_count)
            if count <= 0:
                continue
            if count > INT64_MAX:
                raise OverflowError("token frequency exceeds the compact int64 representation")

            token_id = len(self.token_heads)
            head = 0
            previous_node = 0
            previous_symbol = 0
            for value in token:
                symbol_id = self._get_symbol_id(value)
                node_id = len(self.node_symbol)
                if node_id > UINT32_MAX:
                    raise OverflowError("too many BPE nodes for the compact uint32 representation")
                self.node_symbol.append(symbol_id)
                self.node_next.append(0)
                self.node_edge.append(0)
                if not head:
                    head = node_id
                if previous_node:
                    self.node_next[previous_node] = node_id
                    pair_id = self._get_pair_id(previous_symbol, symbol_id)
                    self._append_edge(pair_id, previous_node, token_id)
                    new_count = self.pair_count[pair_id] + count
                    if new_count > INT64_MAX:
                        raise OverflowError("pair frequency exceeds the compact int64 representation")
                    self.pair_count[pair_id] = new_count
                    self.total_edge_weight += count
                previous_node = node_id
                previous_symbol = symbol_id

            self.token_heads.append(head)
            self.token_weights.append(count)
            self.token_marks.append(0)

    def _affected_tokens(self, pair_id: int) -> list[int]:
        self._mark_stamp += 1
        if self._mark_stamp >= UINT32_MAX:
            self.token_marks = array("I", [0]) * len(self.token_marks)
            self._mark_stamp = 1
        stamp = self._mark_stamp
        affected: list[int] = []
        edge_id = self.pair_head[pair_id]
        while edge_id:
            token_id = self.edge_token[edge_id]
            if self.token_marks[token_id] != stamp:
                self.token_marks[token_id] = stamp
                affected.append(token_id)
            edge_id = self.edge_next[edge_id]
        return affected

    def _release_pair(self, pair_id: int) -> None:
        if self.pair_head[pair_id]:
            raise RuntimeError("attempted to release a pair that still has active edges")
        left = self.pair_left[pair_id]
        right = self.pair_right[pair_id]
        self.pair_map.pop(self._pair_key(left, right), None)
        self.pair_left[pair_id] = 0
        self.pair_right[pair_id] = 0
        self.pair_count[pair_id] = 0
        self.free_pairs.append(pair_id)

    def _apply_deltas(self, deltas: dict[int, int]) -> None:
        for pair_id, delta in deltas.items():
            if not delta:
                continue
            old_count = self.pair_count[pair_id]
            new_count = old_count + delta
            if new_count < 0:
                raise RuntimeError("pair frequency became negative")
            if new_count > INT64_MAX:
                raise OverflowError("pair frequency exceeds the compact int64 representation")

            if old_count > 0:
                self.buckets.remove(pair_id, old_count)
            self.pair_count[pair_id] = new_count
            if new_count > 0:
                self.buckets.add(pair_id, new_count)
            else:
                self._release_pair(pair_id)

    def merge_pair(self, pair_id: int, merged_symbol: int) -> bool:
        left_symbol = self.pair_left[pair_id]
        right_symbol = self.pair_right[pair_id]
        affected = self._affected_tokens(pair_id)
        if not affected:
            return False

        deltas: dict[int, int] = {}
        for token_id in affected:
            weight = self.token_weights[token_id]
            node = self.token_heads[token_id]

            # Remove all old edges for this sequence.  Edge IDs are recycled,
            # and count changes are aggregated once per pair below.
            while node:
                following = self.node_next[node]
                if not following:
                    break
                edge_id = self.node_edge[node]
                if not edge_id:
                    raise RuntimeError("missing edge for an active adjacent node pair")
                self.node_edge[node] = 0
                old_pair_id = self._remove_edge(edge_id)
                deltas[old_pair_id] = deltas.get(old_pair_id, 0) - weight
                node = following

            # Scan left-to-right to retain the BPE rule that overlapping
            # occurrences are merged from the left (for example, aaa -> aa a).
            node = self.token_heads[token_id]
            while node:
                following = self.node_next[node]
                if following and self.node_symbol[node] == left_symbol and self.node_symbol[following] == right_symbol:
                    after_pair = self.node_next[following]
                    self.node_symbol[node] = merged_symbol
                    self.node_next[node] = after_pair
                    self.node_edge[node] = 0
                    self.node_next[following] = 0
                    self.node_edge[following] = 0
                    node = after_pair
                else:
                    node = following

            # Rebuild only this sequence's edges.  The node storage itself is
            # reused, so the number of allocated nodes/edges stays bounded by
            # the initial corpus representation.
            node = self.token_heads[token_id]
            while node:
                following = self.node_next[node]
                if not following:
                    break
                new_pair_id = self._get_pair_id(self.node_symbol[node], self.node_symbol[following])
                self._append_edge(new_pair_id, node, token_id)
                deltas[new_pair_id] = deltas.get(new_pair_id, 0) + weight
                node = following

        self._apply_deltas(deltas)
        return True

    def drop_pair(self, pair_id: int) -> None:
        old_count = self.pair_count[pair_id]
        if old_count > 0:
            self.buckets.remove(pair_id, old_count)
        self.pair_count[pair_id] = 0
        self._release_pair(pair_id)
