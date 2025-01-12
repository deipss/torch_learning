from tqdm import tqdm
import time
if __name__ == '__main__':

    for i in tqdm(range(10), desc="Outer loop", position=0, leave=True):
        for j in tqdm(range(10), desc="Inner loop", position=1, leave=False):
            time.sleep(0.001)
            print(i,j)
