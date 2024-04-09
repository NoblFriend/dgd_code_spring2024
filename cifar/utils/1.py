from itertools import zip_longest

def create(l):
    def f():
        n = l
        while len(n) <= ord(l)-ord('a') + 2:
            yield n
            n += l
    return f

funcs = [create(l)() for l in 'abcdefg']


for param in zip_longest(*[create(l)() for l in 'abcdefg']):
    print(*filter(None, param))