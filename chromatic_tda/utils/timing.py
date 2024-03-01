from time import perf_counter

from chromatic_tda.utils.singleton import singleton


@singleton
class TimingUtils:
    total_time_dict : dict[str, float]
    started_at_dict : dict[str, float]
    call_count_dict : dict[str, int]
    log_times : bool

    def __init__(self, log_times=False):
        self.log_times = log_times
        self.flush()

    def flush(self):
        self.total_time_dict = {}
        self.started_at_dict = {}
        self.call_count_dict = {}

    def start(self, topic: str):
        self.started_at_dict[topic] = perf_counter()
        if self.log_times:
            if topic in self.call_count_dict:
                self.call_count_dict[topic] += 1
            else:
                self.call_count_dict[topic] = 1

    def stop(self, topic: str):
        if self.log_times:
            if topic in self.total_time_dict:
                self.total_time_dict[topic] += perf_counter() - self.started_at_dict[topic]
            else:
                self.total_time_dict[topic] = perf_counter() - self.started_at_dict[topic]

    def print(self):
        for topic in sorted(self.total_time_dict):
            print(f"Time[{topic:<58}] = {self.total_time_dict[topic]:6.2f} s "
                  f"(# calls : {self.call_count_dict[topic]:<6}, "
                  f"average = {self.total_time_dict[topic]/self.call_count_dict[topic]:7.4f})")
