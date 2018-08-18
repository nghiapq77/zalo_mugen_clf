from subprocess import Popen, PIPE, STDOUT
import os
import errno
import csv
from PIL import Image
import eyed3

from config import pixelPerSecond

#Define current path
currentPath = os.path.dirname(os.path.realpath(__file__))

#Remove logs
eyed3.log.setLevel("ERROR")

def isMono(filename):
	audiofile = eyed3.load(filename)
	return audiofile.info.mode == 'Mono'

#Create spectrogram from mp3 files
def createSpectrogram(filename, newfilename, audioPath):
    spectrogramsPath = os.path.join(audioPath, 'spectrograms/')
    #create folder if not exist
    if not os.path.exists(os.path.dirname(spectrogramsPath)):
		try:
			os.makedirs(os.path.dirname(spectrogramsPath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
    
    if not os.path.exists(os.path.join(spectrogramsPath, newfilename[:-4]+".png")): ###check if spectrogram exists alr
        if isMono(audioPath+filename):
		    command = "cp '{}' '/tmp/{}'".format(audioPath+filename,filename)
        else:
	        command = "sox '{}' '/tmp/{}' remix 1,2".format(audioPath+filename,filename)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)

        output, errors = p.communicate()
        if errors:
	        print errors

        #Create spectrogram
        command = "sox '/tmp/{}' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(filename,pixelPerSecond,spectrogramsPath+newfilename[:-4])
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    
        output, errors = p.communicate()
        if errors:
	        print errors

        #Remove tmp mono track
        os.remove("/tmp/{}".format(filename))

def createSpectrogramsFromAudio(audioPath, csvfilename): #also get its genre from csv file
    path = os.path.join(audioPath, csvfilename)
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            newfilename = rows[1]+"_"+rows[0]
            createSpectrogram(rows[0], newfilename, audioPath)

#Create slices of spectrograms
def sliceSpectrogram(specPath, filename, size):
    #get genre
    genre = filename.split("_")[0]
    #load image
    img = Image.open(specPath+filename)
    #compute approx. number of sizexsize samples
    width, height = img.size
    nSamples = int(width/size)
    #create path if not exist
    slicePath = specPath+"slices/{}/".format(genre)
    if not os.path.exists(os.path.dirname(slicePath)):
        try:
            os.makedirs(os.path.dirname(slicePath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    #creating slices
    for i in range(nSamples):
        startPixel = i*size
        imgTmp = img.crop((startPixel, 1, startPixel+size, size+1))
        imgTmp.save(slicePath+"{}_{}.png".format(filename[:-4], i))

def createSlicesFromSpectrograms(specPath, size):
    for filename in os.listdir(specPath):
        if filename.endswith(".png"):
            sliceSpectrogram(specPath, filename, size)