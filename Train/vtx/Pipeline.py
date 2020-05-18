import numpy
import h5py
import random

class Accessor:
    @staticmethod
    def createEmptyBatch():
        return {"X":[],"assoc":[],"y":[],"y_avg":[]}
    
    @staticmethod
    def fileSize(f):
        return f['X'].shape[0]
        
    @staticmethod
    def batchSize(batch):
        return sum(map(lambda x: x.shape[0],batch['X']))
       
    @staticmethod 
    def readFile(batch, f,start,end):
        batch['X'].append(f['X'][start:end])
        batch['assoc'].append(f['assoc'][start:end])
        batch['y'].append(f['y'][start:end])
        batch['y_avg'].append(f['y_avg'][start:end])
        
    @staticmethod
    def concatenateBatch(batch):
        batch['X'] = numpy.concatenate(batch['X'],axis=0)
        batch['assoc'] = numpy.concatenate(batch['assoc'],axis=0)
        batch['y'] = numpy.concatenate(batch['y'],axis=0)
        batch['y_avg'] = numpy.concatenate(batch['y_avg'],axis=0)
        
class Pipeline():
    def __init__(self,inputFileList,testFraction,dataAccessor=Accessor):
        self.inputFileList = inputFileList   
        self.testFraction = testFraction
        self.dataAccessor = dataAccessor
    
    def generate(self,batchSize=100,nFiles=2,isTraining=True,shuffle=True):
        fileIndices = list(range(len(self.inputFileList)))
        readIndex = 0
        if shuffle:
            random.shuffle(fileIndices)
        openFiles = []
        
        def totalsize(fileList):
            n = 0
            for f in fileList:
                n+=f['size']-f['index']
            return n
          
        while readIndex==0 or readIndex<len(self.inputFileList) or totalsize(openFiles)>batchSize:
            while (len(openFiles)<min(len(self.inputFileList),nFiles) or totalsize(openFiles)<batchSize) and readIndex<(len(self.inputFileList)):
                #print 'loading ',inputFileList[fileIndices[readIndex]]
                filePath = self.inputFileList[fileIndices[readIndex]]
                f = h5py.File(filePath,'r')
                if isTraining:
                    openFiles.append({
                        'path':filePath,
                        'handle':f,
                        'index':0,
                        'size':int(self.dataAccessor.fileSize(f)*min(1.,max(0.,1.-self.testFraction)))
                    })
                else:
                    openFiles.append({
                        'path':filePath,
                        'handle':f,
                        'index':int(self.dataAccessor.fileSize(f)*min(1.,max(0.,1.-self.testFraction))),
                        'size':self.dataAccessor.fileSize(f)
                    })
                readIndex+=1
        
            if totalsize(openFiles)>batchSize:  
                batch = self.dataAccessor.createEmptyBatch()
                while (self.dataAccessor.batchSize(batch)<batchSize):
                    currentBatchSize = self.dataAccessor.batchSize(batch)
                    #print '  batch size ',currentBatchSize
                    chosenFileIndex = random.randint(0,len(openFiles)-1)
                    f = openFiles[chosenFileIndex]
                    nread = min([f['size']-f['index'],max(1,int(1.*batchSize/nFiles)),batchSize-currentBatchSize])
                    
                    #print '  reading ',f['path'],f['index'],f['index']+nread,f['size']
                    
                    self.dataAccessor.readFile(batch,f['handle'],f['index'],f['index']+nread)
                    f['index']+=nread
                    if (f['index']+1)>=f['size']:
                        elem = openFiles.pop(chosenFileIndex)
                        #print 'dequeue ',elem,len(openFiles)
                    
                self.dataAccessor.concatenateBatch(batch)
                yield batch
            
            
            
