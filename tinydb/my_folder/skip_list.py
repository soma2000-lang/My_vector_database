from __future__ import annotations

import random


class Node:
    def __init__(self, value: int, level: int) -> None:
        self.value = value
        self.pointers = [None for _ in range(level + 1)]

    def __repr__(self) -> str:
        return str(self.value)
    def __init__(
        self, lst: list[int] | None = None, max_level: int = 2, p: float = 0.5
    ) -> None:
        assert max_level >= 0

        self.max_level = max_level  # note: max_level is 0-indexed (0 means 1 level, 1 means 2 levls, etc.)
        self.level = 0
        self.p = p
        self.header = Node(value=-1, level=self.max_level)

        if lst is None:
            lst = []

        for value in lst:
            self.insert(value)
    def _random_level(self) -> int:
            """
        Coin-flipping implementation for sampling -- we could make this more
        efficient using the alias method.
        """
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level