import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today) # 값이 변할 때마다 시간 변경 확인 가능

model = st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4"
    ),
)

# st.write(model)

if model == "GPT-3":
    st.write("cheap")
else:
    st.write("not cheap")
    name = st.text_input("What is your name?")
    st.write(name) # enter key -> refresh

    value = st.slider("temperature", min_value=0.1, max_value=1.0)
    st.write(value)