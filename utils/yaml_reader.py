import yaml


class YamlUtil:
    _cache = {}

    @staticmethod
    def read(file_path: str) -> dict:
        if file_path in YamlUtil._cache:
            return YamlUtil._cache[file_path]

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        YamlUtil._cache[file_path] = data
        return data

    @staticmethod
    def read_from_string(yaml_str: str) -> dict:
        return yaml.safe_load(yaml_str)

    @staticmethod
    def clear_cache(file_path: str = None):
        if file_path:
            YamlUtil._cache.pop(file_path, None)
        else:
            YamlUtil._cache.clear()
