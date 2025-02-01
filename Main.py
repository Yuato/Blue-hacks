# import required libraries
import numpy as np
from numpy.linalg import norm
import util as util
import csv

class element():
    def __init__(self, name, file):
        self.name = name
        self.likely = ("",0)
        self.f = util.load_data(file, "Intensity")
        print(self.f)

    def test(self):
        print(1)
        A = np.array(util.stringtofloat(self.f))
        elements = util.load_data("files.csv","Names")

        #convert and test all elements using cosine similarity
        for name in elements:
            test = util.load_data("elements/"+name+".csv","Intensity")
            
            B = np.array(util.stringtofloat(test))
        
            #compute cosine similarity
            cosine = util.cosine_similarity(A,B)
            print("Cosine Similarity:", cosine)

            if (cosine>self.likely[1]):
                self.likely = (name, cosine)

    def identify(self):
        return self.likely

def main():
    eth = element("eth", "elements/ethanol.csv")
    eth.test()
    print(eth.identify())

main()