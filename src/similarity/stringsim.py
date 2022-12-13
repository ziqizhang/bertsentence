import Levenshtein

class StringSim:
    def __init__(self, algorithm):
        pass

    def similarity(self, s1, s2):
        return Levenshtein.ratio(s1, s2)


