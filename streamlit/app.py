import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import math
import altair as alt

@st.cache_data
def send_audio(file):
    print("############################################")
    print('attempting to send audio')
    url = 'http://flasknginx:8080/sound/'

    files = {'soundFile': file}
    res = requests.post(url, files=files)
    # res.content # type : bytes
    print("contenttype", type(res.content))
    # print(type(res))
    # print(res.status_code)

    if res.ok:
        resContent = res.json() # type : dict 
        # print(resContent)
        # print(type(resContent))
        print("Upload completed successfully!")
        st.write("Upload completed successfully!")

        return resContent
    else:
        print("Something went wrong!")
        st.write("Something went wrong!")
        st.stop()

@st.cache_data
def draw_fig(data, threshold):

     # グラフ描画
    Data_AnomalyScore = pd.DataFrame({
        "AnomalyScore":data,
        "Time": np.arange(len(data))/60
        })
    # print(Data_AnomalyScore)
   
    timeBatch_num=3600 # 一時間ごとにグラフを描画
    TIMEBATCH_INDICES = np.arange(start=0, stop=len(data), step=timeBatch_num)  
    TIMEBATCH_INDICES = np.append(TIMEBATCH_INDICES, len(data)) 
    # print("TIMEBATCH_INDICES:", TIMEBATCH_INDICES)

    for index in np.arange(len(TIMEBATCH_INDICES) - 1):
        timeBatch_start = TIMEBATCH_INDICES[index]  
        timeBatch_end = TIMEBATCH_INDICES[index + 1] 
        
        # clip=True : 「.scale」で指定したdomainの範囲外の値を消す
        cart = alt.Chart(Data_AnomalyScore.iloc
        [timeBatch_start:timeBatch_end]).mark_line(clip=True).encode(
            alt.Y("AnomalyScore").scale(domain=(threshold, 0.006)),
            alt.X("Time").title('Time[min]'),
            
        ).properties(
        height=300, width=500
        )
        st.altair_chart(cart, use_container_width=True)

def main():
    st.title('異常度可視化')
    st.subheader('wavファイル選択')
    file = st.file_uploader('Upload image file', type=["wav"])
    # print("type(file)",type(file))
    if file :
        resContent = send_audio(file)
        notburied_anomary_scores =np.array(resContent["predictions"]) 
        # print(len(notburied_anomary_scores))

        # 閾値の設定
        min = np.unique(notburied_anomary_scores)[1] # 異常度の最小値(0以外を取得するために[1]を指定)
        min = math.floor(min * 10000) / 10000 # 小数点第五位を切り捨て
        # print("min:", min)

        options=[]
        for i in range(0, 60):
            value = i*0.0001
            x =  math.floor(value * 10000) / 10000 
            options.append(x)
        options = list(filter(lambda x: x >= min, options))
        threshold = st.select_slider(
        '閾値',options=options, value=min)
        draw_fig(notburied_anomary_scores, threshold)

if __name__ == "__main__":
    main()

