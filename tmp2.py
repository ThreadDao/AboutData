import random
import pandas as pd

def f(pk_index):
    length = random.randint(1, 10)
    print(length)
    print(f"from {pk_index} to {pk_index + length}")
    return length


if __name__ == '__main__':
   a = 5000000000
   print(a+1)
