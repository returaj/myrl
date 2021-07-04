import yaml


class Struct:
    def __init__(self, dataMap):
        for name, value in dataMap.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


class Configuration:
    @staticmethod
    def construct(path):
        with open(path, 'r') as fp:
            dataMap = yaml.safe_load(fp)
        return Struct(dataMap)

