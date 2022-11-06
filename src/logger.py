import sys
import os

class Logger(object):
    def __init__(self,path='./',filename='train.log'):
        super(Logger).__init__()
        self.terminal = sys.stdout
        self.log = open(os.path.join(path,filename),mode='a')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()