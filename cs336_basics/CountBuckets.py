from array import array

class CountBuckets:

    __slots__ = (
            "heads",
            "best",
            "levels",
            "pair_left",
            "pair_right",
            "pair_prev",
            "pair_next",
            "symbol_bytes",
        )
    
    def __init__(
        self,
        pair_left: array,
        pair_right: array,
        pair_prev: array,
        pair_next: array,
        symbol_bytes: list[bytes],
        count_limit: int
    ) -> None:
        
        self.heads: dict[int, int] = {}
        self.best: dict[int, int] = {}
        # Each level stores 64 child-presence bits per sparse integer block.
        level_count = max(1, (max(0, count_limit).bit_length() + 5) // 6)
        self.levels: list[dict[int, int]] = [{} for _ in range(level_count)]
        self.pair_left = pair_left
        self.pair_right = pair_right
        self.pair_prev = pair_prev
        self.pair_next = pair_next
        self.symbol_bytes = symbol_bytes

    def _ensure_capacity(self, count: int) -> None:
        while count >= (1 << (6 * len(self.levels))):
            old_top = self.levels[-1]
            self.levels.append({})
            new_top = self.levels[-1]
            for child in old_top:
                parent = child >> 6
                new_top[parent] = new_top.get(parent, 0) | (1 << (child & 63))

    def _set_occupied(self, count: int) -> None:
        self._ensure_capacity(count)
        child = count >> 6
        bit = 1 << (count & 63)
        old_mask = self.levels[0].get(child, 0)
        if old_mask & bit:
            return
        self.levels[0][child] = old_mask | bit
        if old_mask:
            return

        for level in range(1, len(self.levels)):
            parent = child >> 6
            bit = 1 << (child & 63)
            old_mask = self.levels[level].get(parent, 0)
            self.levels[level][parent] = old_mask | bit
            if old_mask:
                return
            child = parent

    def _clear_occupied(self, count: int) -> None:
        child = count >> 6
        bit = 1 << (count & 63)
        new_mask = self.levels[0][child] & ~bit
        if new_mask:
            self.levels[0][child] = new_mask
            return
        del self.levels[0][child]

        for level in range(1, len(self.levels)):
            parent = child >> 6
            bit = 1 << (child & 63)
            new_mask = self.levels[level][parent] & ~bit
            if new_mask:
                self.levels[level][parent] = new_mask
                return
            del self.levels[level][parent]
            child = parent

    def _highest_count(self) -> int | None:
        top_level = len(self.levels) - 1
        if top_level == 0:
            mask = self.levels[0].get(0, 0)
            if not mask:
                return None
            return mask.bit_length() - 1

        # count_limit ensures that the highest-level key is normally zero.
        # The fallback also makes the structure safe if it grows dynamically.
        top_map = self.levels[top_level]
        parent = 0
        mask = top_map.get(parent, 0)
        if not mask:
            if not top_map:
                return None
            parent = max(top_map)
            mask = top_map[parent]
        child = (parent << 6) | (mask.bit_length() - 1)

        for level in range(top_level - 1, -1, -1):
            mask = self.levels[level].get(child, 0)
            if not mask:
                return None
            child = (child << 6) | (mask.bit_length() - 1)
        return child

    def _pair_greater(self, first: int, second: int) -> bool:
        first_left = self.symbol_bytes[self.pair_left[first]]
        second_left = self.symbol_bytes[self.pair_left[second]]
        if first_left != second_left:
            return first_left > second_left
        return self.symbol_bytes[self.pair_right[first]] > self.symbol_bytes[self.pair_right[second]]

    def add(self, pair_id: int, count: int) -> None:
        old_head = self.heads.get(count, 0)
        self.pair_prev[pair_id] = 0
        self.pair_next[pair_id] = old_head
        if old_head:
            self.pair_prev[old_head] = pair_id
        else:
            self._set_occupied(count)
        self.heads[count] = pair_id

        old_best = self.best.get(count, 0)
        if old_best and self._pair_greater(pair_id, old_best):
            self.best[count] = pair_id
        elif not old_best and not old_head:
            self.best[count] = pair_id

    def remove(self, pair_id: int, count: int) -> None:
        previous = self.pair_prev[pair_id]
        following = self.pair_next[pair_id]
        if previous:
            self.pair_next[previous] = following
        else:
            self.heads[count] = following
        if following:
            self.pair_prev[following] = previous

        self.pair_prev[pair_id] = 0
        self.pair_next[pair_id] = 0
        if self.best.get(count) == pair_id:
            self.best.pop(count, None)
        if not following and not previous:
            self.heads.pop(count, None)
            self._clear_occupied(count)

    def move(self, pair_id: int, old_count: int, new_count: int) -> None:
        if old_count == new_count:
            return
        if old_count > 0:
            self.remove(pair_id, old_count)
        if new_count > 0:
            self.add(pair_id, new_count)

    def best_pair(self) -> int:
        count = self._highest_count()
        if count is None:
            return 0

        pair_id = self.best.get(count, 0)
        if pair_id:
            return pair_id

        # A bucket is normally cached as pairs enter it.  Recompute only when
        # its cached maximum was the pair that just moved out.
        pair_id = self.heads[count]
        candidate = pair_id
        pair_id = self.pair_next[pair_id]
        while pair_id:
            if self._pair_greater(pair_id, candidate):
                candidate = pair_id
            pair_id = self.pair_next[pair_id]
        self.best[count] = candidate
        return candidate