import numpy

class FastHisto():
    def __init__(
        self,
        nbins=256,
    ):
        self.nbins = nbins
        self.max_z0 = 20.46912512
            
    def predictZ0(self,value,weight):
        z0List = []
        halfBinWidth = 0.5*(2*self.max_z0)/self.nbins
        for ibatch in range(value.shape[0]):
            hist,bin_edges = numpy.histogram(value[ibatch],self.nbins,range=(-1*self.max_z0,self.max_z0),weights=weight[ibatch])
            hist = numpy.convolve(hist,[1,1,1],mode='same')
            z0Index= numpy.argmax(hist)
            z0 = -1*self.max_z0+(2*self.max_z0)*z0Index/self.nbins+halfBinWidth
            z0List.append([z0])
        return numpy.array(z0List,dtype=numpy.float32)
