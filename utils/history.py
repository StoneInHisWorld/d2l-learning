class History:
    def __init__(self, *args: str):
        self.__keys__ = args
        for k in args:
            self.__setattr__(k, [])

    def __getitem__(self, key):
        return getattr(self, key)

    def add(self, keys, values):
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            self[k].append(v)

    def __iter__(self):
        for k in self.__keys__:
            yield k, self[k]

    def __str__(self):
        ret = ''
        for k, v in self:
            ret += k + str(v)
        return ret
