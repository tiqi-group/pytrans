import re
re_ix = r"(\[[\s\d,]+\]|[-\d:]+)"


def parse_indexing_string(indexing: str):
    """Parse a numpy-style indexing string to a tuple of slices
    freely inspired from https://stackoverflow.com/a/43090200

    Args:
      indexing: str: can contain digits, commas, square brackets, colons, and whitespace

    Returns:
      slices: tuple of slice, int, or list of int

    Examples:
      '[0, 1], 10, ::-1' -> ([0, 1], 10, slice(None, None, -1))
      ':, :, 1' -> (slice(None, None, None), slice(None, None, None), 1)
      '0:-4:10,::-1,112,[0,10,18]' -> (slice(0, -4, 10), slice(None, None, -1), 112, [0, 10, 18])
      ':, 3,...,10' -> invalid string
      ':, a, 11' -> invalid string
      '[:,1], ::' -> invalid string
      '1, 3, [, 11' -> invalid string
    """
    rem = re.sub(re_ix, '', indexing)
    if not set(rem) <= {',', ' '}:
        # there should be no other characters left after matching
        raise ValueError("invalid string")
    slices = []
    for match in re.findall(re_ix, indexing):
        if '[' in match:
            sl = list(map(int, match.strip('[]').split(',')))
        elif ':' in match:
            sl = slice(*(int(i) if i else None for i in match.strip().split(':')))
        else:
            sl = int(match)
        slices.append(sl)
    return tuple(slices)


def parse_indexing(indexing):
    if isinstance(indexing, (slice, list, int)):
        return indexing
    elif isinstance(indexing, str):
        return parse_indexing_string(indexing)
    else:
        raise TypeError("Incorrect type of indexing")


if __name__ == '__main__':
    strings = """[0, 1], 10, ::-1
:, :, 1
0:-4:10,::-1,112,[0,10,18]
:, 3,...,10
:, a, 11
[:,1], ::
1, 3, [, 11""".split('\n')

    for string in strings:
        print(f"'{string}'", end=' -> ')
        try:
            print(parse_indexing_string(string))
        except ValueError as e:
            print(e)
