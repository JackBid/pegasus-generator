class Util():

    def __init__(self):
        return
    
    # helper function to make getting another batch of data easier
    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

