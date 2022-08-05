import cv2
import numpy as np
import os
import streamlit as st
from recongnizer import maincall, train
from spot_diff import spot_diff
from datetime import datetime
import time



st.title("Minor Project")
option = st.selectbox(
    'How would you like to be contacted?',
    ('Select the function', 'Monitor', 'Visitor Check Mode', 'Motion Test', 'Face Recognition', 'Capture Video'))

FRAME_WINDOW = st.image([])
FRAME_WINDOW_2 = st.image([])
FRAME_WINDOW_3 = st.image([])
cap = cv2.VideoCapture(0)

if option == 'Face Recognition':
    col1, col2, col3= st.columns([1,3,1])
    with col2:
        perform = st.selectbox(
        'Wether to Detect or Add Member into the Model',
        ('Select', 'Detect', 'Add Member'))
    filename = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(filename)
    if perform=='Detect':
	    recog, labelslist = maincall()
    while perform=='Detect':
        _, frm = cap.read()
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]

            label = recog.predict(roi)
            # print(label)
            if label[1] < 100:
                cv2.putText(frm, f"{labelslist[(str(label[0]))]} + {int(label[1])}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                cv2.putText(frm, "unkown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # cv2.imshow("identify", frm)
        FRAME_WINDOW.image(frm)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
    
    if perform=='Add Member':
        collect_boolean = True
        count = 1
        filename = "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(filename)
        # cap = cv2.VideoCapture(0)
        name = st.text_input("Enter the name", "Type Here")
        ids = st.text_input("ID", "Type Here")
        if st.button("Submit"):
            while collect_boolean:
                ret, frm = cap.read()
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.4, 1)
                for x, y, w, h in faces:
                    cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 2)
                    roi = gray[y:y+h, x:x+w]
                    cv2.imwrite(f"persons\\{name}-{count}-{ids}.jpg", roi)
                    count = count + 1
                    cv2.putText(frm, f"{count}", (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
                    FRAME_WINDOW.image(roi)
                    # cv2.imshow("new", roi)

                FRAME_WINDOW.image(frm)


                # cv2.imshow("identify", frm)
                
                if cv2.waitKey(1) == 27 or count > 300:
                    cv2.destroyAllWindows()
                    cap.release()
                    collect_boolean = False
                    train()
                    break

if option == 'Monitor':
	spot = st.checkbox("Spot Difference")
	if spot:
		motion_detected = False
		is_start_done = False
		cap = cv2.VideoCapture(0)
		check = []
		print("waiting for 2 seconds")
		time.sleep(2)
		frame1 = cap.read()
		_, frm1 = cap.read()
		frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)
	while spot:
		_, frm2 = cap.read()
		frm2 = cv2.cvtColor(frm2, cv2.COLOR_BGR2GRAY)
		diff = cv2.absdiff(frm1, frm2)
		_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
		contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		#look at it
		contors = [c for c in contors if cv2.contourArea(c) > 25]
		if len(contors) > 5:
			cv2.putText(thresh, "motion detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
			motion_detected = True
			is_start_done = False

		elif motion_detected and len(contors) < 3:
			if (is_start_done) == False:
				start = time.time()
				is_start_done = True
			end = time.time()

			print(end-start)
			if (end - start) > 4:
				frame2 = cap.read()
				cap.release()
				cv2.destroyAllWindows()
				x = spot_diff(frame1, frame2, FRAME_WINDOW)
				if x == 0:
					print("running again")
					break

				else:
					print("found motion")
					break

		else:
			cv2.putText(thresh, "no motion detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

		# cv2.imshow("winname", thresh)
		FRAME_WINDOW_2.image(thresh)
		_, frm1 = cap.read()
		frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

		if cv2.waitKey(1) == 27:
			break

if option == 'Visitor Check Mode':
    cap = cv2.VideoCapture(1)
    right, left = "", ""
    check = st.checkbox("Check Mode")
    while check:
        _, frame1 = cap.read()
        frame1 = cv2.flip(frame1, 1)
        _, frame2 = cap.read()
        frame2 = cv2.flip(frame2, 1)
        diff = cv2.absdiff(frame2, frame1)        
        diff = cv2.blur(diff, (5,5))        
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)        
        _, threshd = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contr, _ = cv2.findContours(threshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        x = 300

        if len(contr) > 0:
            max_cnt = max(contr, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(max_cnt)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame1, "MOTION", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            
        if right == "" and left == "":
            if x > 500:
                right = True
            elif x < 200:
                left = True
                
        elif right:
                if x < 200:
                    print("to left")
                    x = 300
                    right, left = "", ""
                    cv2.imwrite(f"visitors/in/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", frame1)
            
        elif left:
                if x > 500:
                    print("to right")
                    x = 300
                    right, left = "", ""
                    cv2.imwrite(f"visitors/out/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", frame1)

        # cv2.imshow("", frame1)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame1)
        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

if option == 'Motion Test':
    testing = st.checkbox("Motion test")
    while testing:
        _, frame1 = cap.read()
        _, frame2 = cap.read()

        diff = cv2.absdiff(frame2, frame1)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        diff = cv2.blur(diff, (5,5))
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        contr, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contr) > 0:
            max_cnt = max(contr, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(max_cnt)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame1, "Motion", (200,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

        else:
            cv2.putText(frame1, "No Motion", (200,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        # cv2.imshow("esc. to exit", frame1)
        FRAME_WINDOW.image(frame1)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

    # ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # FRAME_WINDOW.image(frame)

if option == 'Capture Video':
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'Recordings/{datetime.now().strftime("%H-%M-%S")}.avi', fourcc,20.0,(640,480))
    col1, col2, col3= st.columns([1,10,1])
    with col2:
        record = st.checkbox('Record')
    while record:
        _, frame = cap.read()
        cv2.putText(frame, f'{datetime.now().strftime("%D-%H-%M-%S")}', (50,50), cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (255,255,255), 2)
        out.write(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("esc. to stop", frame)
        FRAME_WINDOW.image(frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break 
