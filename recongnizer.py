import os
import cv2
import numpy as np

def maincall():
    paths = [os.path.join("persons", im) for im in os.listdir("persons")]
    labelslist = {}
    for path in paths:
        print(path)
        labelslist[path.split('\\')[-1].split('-')[2].split('.')
                            [0]] = path.split('\\')[-1].split('-')[0]

    print(labelslist)
    recog = cv2.face.LBPHFaceRecognizer_create()

    recog.read('model.yml')
    return recog, labelslist

def train():
	print("training part initiated !")

	recog = cv2.face.LBPHFaceRecognizer_create()

	dataset = 'persons'

	paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]

	faces = []
	ids = []
	labels = []
	for path in paths:
		labels.append(path.split('\\')[-1].split('-')[0])

		ids.append(int(path.split('\\')[-1].split('-')[2].split('.')[0]))

		faces.append(cv2.imread(path, 0))

	recog.train(faces, np.array(ids))

	recog.save('model.yml')

	return