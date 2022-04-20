from itertools import islice
from random import randint


def random_chunk(li, min_chunk=1, max_chunk=3):
    it = iter(li)
    while True:
        nxt = list(islice(it, randint(min_chunk, max_chunk)))
        if nxt:
            yield nxt
        else:
            break