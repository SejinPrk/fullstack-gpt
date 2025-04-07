import streamlit as st
from langchain.prompts import PromptTemplate

st.write("hello")

# st.write([1, 2, 3, 4])
a = [1, 2, 3, 4]
a

# st.write({"x": 1})
d =  {"x": 1}
d

# st.write(PromptTemplate)
PromptTemplate

p = PromptTemplate.from_template("xxxx")

# st.write(p)
p

st.selectbox(
    "Choose your model",
    ("GPT-3", "GPT-4")
)