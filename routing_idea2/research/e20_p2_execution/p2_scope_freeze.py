"""E18 read-only JSON containers reused for a single P2 scope snapshot."""

def _deny(*args, **kwargs):
    raise TypeError("An approved scope snapshot is read-only")


class FrozenDict(dict):
    """JSON-compatible dictionary with ordinary mutation operations disabled."""

    def __init__(self, *args, **kwargs):
        if getattr(self, "_sealed", False):
            _deny()
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_sealed", True)

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _deny
    __ior__ = __setattr__ = __delattr__ = _deny

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def copy(self):
        return self


class FrozenList(list):
    """JSON-compatible list with ordinary mutation operations disabled."""

    def __init__(self, *args):
        if getattr(self, "_sealed", False):
            _deny()
        list.__init__(self, *args)
        object.__setattr__(self, "_sealed", True)

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = _deny
    reverse = sort = __iadd__ = __imul__ = __setattr__ = __delattr__ = _deny

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def copy(self):
        return self


def freeze_scope(value):
    """Copy a JSON tree into read-only JSON-compatible dict/list subclasses."""
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: freeze_scope(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_scope(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("Scope snapshots accept JSON values only")


