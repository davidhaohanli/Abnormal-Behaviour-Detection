class Spliter(object):
    def __init__(self,discardFloor,splitCeil,fullFillGate):
        self.floor=discardFloor;
        self.ceil=splitCeil;
        self.gate=fullFillGate
    def split(self,pos):
        