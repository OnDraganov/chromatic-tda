import multiprocessing
import math
import os

from multiprocessing.managers import SharedMemoryManager
from chromatic_tda.utils.timing import TimingUtils

N = 5*1000*1000

class ParallelTestCore():
    def __init__(self):
        self.atoms = os.cpu_count()
        self.actions = math.ceil(N/self.atoms)

    def cube(self, x):
        return math.sqrt(x)*math.sqrt(x)*math.sqrt(x)*math.sqrt(x)*math.sqrt(x)*math.sqrt(x)*math.sqrt(x)

class LinearTest(ParallelTestCore):
    def run(self):
        TimingUtils().start("Linear")
        result = {}
        for x in range(0, N):
            result[x] = self.cube(x)
        print(len(result))
        print(result[int(N/2)])
        TimingUtils().stop("Linear")

class ParallelWithPoolTest(ParallelTestCore):

    def cube_atoms(self, x):
        print(f"atom {x}")
        res = []
        for i in range(0, self.actions):
            k = x * self.actions + i
            if k >= N:
                break
            res.append(self.cube(k))
        return res        

    def run(self):
        TimingUtils().start("Pool")
        result = []
        with multiprocessing.Pool() as pool:
            res_list = pool.map(self.cube_atoms, range(0, self.atoms))
            for res in res_list:
                result.extend(res)
        print(len(result))
        #print(result[int(N/2)])
        TimingUtils().stop("Pool")


class ParallelWithProcessTest(ParallelTestCore):

    def cube_atoms(self, x, res):
        result = {}
        for i in range(0, self.actions):
            k = x * self.actions + i
            if k >= N:
                break
            result[k] = self.cube(k)
        res[x] = result

    def run(self):
        TimingUtils().start("Process")
        with SharedMemoryManager() as smm:
            process_dict = {}
            result = {}
            process_res = smm.ShareableList(range(self.atoms))
            for x in range(0, self.atoms):
                process = multiprocessing.Process(target= self.cube_atoms, args=[x, process_res])
                process.start()            
                process_dict[x] = process

            for x in range(0, self.atoms):
                process = process_dict[x]
                process.join()
        
                result = result | process_res[x]
    
        print(len(result))
        TimingUtils().stop("Process")


def parallel():
    LinearTest().run()
    ParallelWithPoolTest().run()
#    ParallelWithProcessTest().run()
    TimingUtils().print()

if __name__ == "__main__":
    parallel()