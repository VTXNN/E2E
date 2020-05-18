import os
import logging
import sys
import glob

class InputFiles():
    def __init__(self,inputFilePatternList):
        self.inputFiles = []
        for pattern in inputFilePatternList:
            matchedFiles = glob.glob(pattern)
            if len(matchedFiles)==0:
                logging.critical("No file matched by pattern: "+pattern)
                sys.exit(1) 
            for matchedFile in matchedFiles:
                absPath = os.path.abspath(matchedFile)
                logging.debug("Adding input file: "+absPath)
                self.inputFiles.append(absPath)
                
        #remove duplicates
        self.inputFiles = list(set(self.inputFiles))
                
        logging.info("Found %i input files"%(len(self.inputFiles)))
       
    def __iter__(self):
        return iter(self.inputFiles)
        
    def __len__(self):
        return len(self.inputFiles)
        
    def __getitem__(self,index):
        return self.inputFiles[index]
