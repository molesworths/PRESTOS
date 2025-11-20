

def batch(iterable, n=1): # TODO
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

def unbatch(batched_list): # TODO
    return [item for batch in batched_list for item in batch]