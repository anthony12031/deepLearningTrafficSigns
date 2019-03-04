# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:36:06 2019

@author: tony_
"""

from keras.models import load_model
import  cv2
import numpy as np
import matplotlib.pyplot as plt
import time

modelo = load_model('traffic_model.h5')

ALTO = 32
ANCHO = 32
CHANNELS = 3

def capturarCamara():
    ultima_prediccion = ""
    cap = cv2.VideoCapture(0)
#    imagenes_predecir=[]
#    num_img=10
    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        
        # Display the resulting frame
        cv2.putText(frame, "Prediccion: {}".format(ultima_prediccion), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #reescalar la imagen
        frame = cv2.resize(frame, (ALTO, ANCHO), interpolation=cv2.INTER_CUBIC)
        time.sleep(0.1)
        ultima_prediccion = predecir(frame)
       
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    

def predecir(imagen):
    prediccion=modelo.predict(np.array(imagen).reshape(-1,32,32,3), verbose=0)
    #print(prediccion)
    clasePredecida=np.argmax(prediccion)
    label = ""
    #print(clasePredecida)
    if(clasePredecida==0):
        label="20 km/h"
        print(label)
    if(clasePredecida==1):
        label="Prohibido adelantar"
        print(label)
    if(clasePredecida==2):
        label="Cruce Peatonal"
        print(label)
    if(clasePredecida==3):
        label="Giro"
        print(label)
    return label
    
    

capturarCamara()

#prueba = plt.imread("test_alemania/2/1.jpg")
#prueba= cv2.resize(prueba, (ALTO, ANCHO), interpolation=cv2.INTER_CUBIC)
#prediccion=modelo.predict(np.array(prueba).reshape(-1,32,32,3), verbose=1)
#print(np.argmax(prediccion))
#
#plt.imshow(prueba)
#plt.show()