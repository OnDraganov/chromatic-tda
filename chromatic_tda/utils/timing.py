from time import perf_counter

from chromatic_tda.utils.singleton import singleton


@singleton
class TimingUtils:
    total_time_dict : dict[str, float]
    started_at_dict : dict[str, float]
    call_count_dict : dict[str, int]

    def __init__(self):
        self.flush()

    def flush(self):
        self.total_time_dict = {}
        self.started_at_dict = {}
        self.call_count_dict = {}

    def start(self, topic: str):
        self.started_at_dict[topic] = perf_counter()
        if topic in self.call_count_dict:
            self.call_count_dict[topic] += 1
        else:
            self.call_count_dict[topic] = 1

    def stop(self, topic: str):
        if topic in self.total_time_dict:
            self.total_time_dict[topic] += perf_counter() - self.started_at_dict[topic]
        else:
            self.total_time_dict[topic] = perf_counter() - self.started_at_dict[topic]

    def print(self):
        for topic in self.total_time_dict:
            print(f"Time [ {topic:<40} ] = {round(self.total_time_dict[topic], 2):<5} seconds (# calls : {self.call_count_dict[topic]:<6}, average = {round(self.total_time_dict[topic]/self.call_count_dict[topic], 3):<6})")
