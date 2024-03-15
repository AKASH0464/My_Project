import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
import mediapipe as mp
from keras.models import load_model

def app():

    df = pd.read_csv('muse_v3.csv')
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils


    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']

    df = df[['name','emotional','pleasant','link','artist']]

    df = df.sort_values(by=["emotional", "pleasant"])
    df.reset_index()

    df_sad = df[:18000]
    df_fear = df[18000:36000]
    df_anger = df[36000:54000]
    df_neutral = df[54000:72000]
    df_happy = df[72000:]

    def fun(list):
        data = pd.DataFrame()
        if len(list) == 1:
            v = list[0]
            t = 30
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)])
            elif v == 'Angry':
                data = pd.concat([data, df_anger.sample(n=t)])
            elif v == 'Fearful':
                data = pd.concat([data, df_fear.sample(n=t)])
            elif v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)])
            else:
                data = pd.concat([data, df_sad.sample(n=t)])

        elif len(list) == 2:
            times = [20,10]

            for i in range (len(list)):
                v = list[i]
                t = times[i]

                if v == 'Neutral':
                    data = pd.concat([data, df_neutral.sample(n=t)])
                elif v == 'Angry':
                    data = pd.concat([data, df_anger.sample(n=t)])
                elif v == 'Fearful':
                    data = pd.concat([data, df_fear.sample(n=t)])
                elif v == 'Happy':
                    data = pd.concat([data, df_happy.sample(n=t)])
                else:
                    data = pd.concat([data, df_sad.sample(n=t)])
        
        elif len(list) == 3:
            times = [15,10,5]

            for i in range (len(list)):
                v = list[i]
                t = times[i]

                if v == 'Neutral':
                    data = pd.concat([data, df_neutral.sample(n=t)])
                elif v == 'Angry':
                    data = pd.concat([data, df_anger.sample(n=t)])
                elif v == 'Fearful':
                    data = pd.concat([data, df_fear.sample(n=t)])
                elif v == 'Happy':
                    data = pd.concat([data, df_happy.sample(n=t)])
                else:
                    data = pd.concat([data, df_sad.sample(n=t)])

        elif len(list) == 4:
            times = [10,9,8,3]

            for i in range (len(list)):
                v = list[i]
                t = times[i]

                if v == 'Neutral':
                    data = pd.concat([data, df_neutral.sample(n=t)])
                elif v == 'Angry':
                    data = pd.concat([data, df_anger.sample(n=t)])
                elif v == 'Fearful':
                    data = pd.concat([data, df_fear.sample(n=t)])
                elif v == 'Happy':
                    data = pd.concat([data, df_happy.sample(n=t)])
                else:
                    data = pd.concat([data, df_sad.sample(n=t)])

        else:
            times = [10,7,6,5,2]

            for i in range (len(list)):
                v = list[i]
                t = times[i]

                if v == 'Neutral':
                    data = pd.concat([data, df_neutral.sample(n=t)])
                elif v == 'Angry':
                    data = pd.concat([data, df_anger.sample(n=t)])
                elif v == 'Fearful':
                    data = pd.concat([data, df_fear.sample(n=t)])
                elif v == 'Happy':
                    data = pd.concat([data, df_happy.sample(n=t)])
                else:
                    data = pd.concat([data, df_sad.sample(n=t)])
        return data

    def pre(l):
        result = [item for items, c in Counter(l).most_common()
                for item in [items] * c]
        
        ul = []

        for x in result:
            if x not in ul:
                ul.append(x)
        print(f"No duplicate emotion: {ul}")
        return ul


    model = load_model("model.h5")
    label = np.load("labels.npy")


    emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad"}

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)

    st.markdown("<h2 style='text-align: center; color: white;'><b>Emotion based music Recommendation</b></h2>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>",unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    list = []
    with col1:
        pass
    with col2:
        if st.button('SCAN EMOTION (Click here)'):
            count = 0
            list.clear()

            while True:
                lst = []
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                res = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if res.face_landmarks:
                    for i in res.face_landmarks.landmark:
                        lst.append(i.x - res.face_landmarks.landmark[1].x)
                        lst.append(i.y - res.face_landmarks.landmark[1].y)

                    if res.left_hand_landmarks:
                        for i in res.left_hand_landmarks.landmark:
                            lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                            lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                    else:
                        for i in range(42):
                            lst.append(0.0)

                    if res.right_hand_landmarks:
                        for i in res.right_hand_landmarks.landmark:
                            lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                            lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                    else:
                        for i in range(42):
                            lst.append(0.0)

                    lst = np.array(lst).reshape(1,-1)
                    pred = label[np.argmax(model.predict(lst))]
                    list.append(pred)
                    cv2.putText(frame, pred, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)



                drawing.draw_landmarks(frame, res.face_landmarks, holistic.FACEMESH_CONTOURS)
                drawing.draw_landmarks(frame, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
                drawing.draw_landmarks(frame, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
                cv2.imshow("window", frame)

                if cv2.waitKey(1) & 0xFF == ord('x'):
                    cap.release()
                    cv2.destroyAllWindows()                
                    break
                #if count>=20:
                #  break

            print(f"with duplicate emotion: {list}")
            list = pre(list)

    with col3:
        pass

    new_df = fun(list)

    st.write("")
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>",unsafe_allow_html=True)

    st.write("--------------------------------------------------------------------------------------------------------------------")

    try:
        for l, a, n, i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):
            st.markdown("<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>""".format(l,i+1,n),unsafe_allow_html=True)

            st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>".format(a),unsafe_allow_html=True)

            st.write("--------------------------------------------------------------------------------------------------------------------")
    except:
        pass
