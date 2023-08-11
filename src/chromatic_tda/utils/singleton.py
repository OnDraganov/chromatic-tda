def singleton(cls):
    __INSTANCES = {}

    def get_instance(*args, **kwargs):
        if cls not in __INSTANCES:
            __INSTANCES[cls] = cls(*args, **kwargs)
        return __INSTANCES[cls]

    return get_instance
