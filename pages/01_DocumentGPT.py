import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📜"
)
st.title('DocumentGPT')

# # 예제1
# with st.chat_message("human"):
#     st.write("Hello")
#
# with st.chat_message("ai"):
#     st.write("How are you?")
#
# with st.status("Embedding file...", expanded=True) as status:
#     time.sleep(3)
#     st.write("Getting the file")
#     time.sleep(3)
#     st.write("Embedding the file")
#     time.sleep(3)
#     st.write("Caching the file")
#     status.update(label="Error", state="error")


# messages = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# st.write(st.session_state["messages"])

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
        if save:
            st.session_state["messages"].append({"message":message, "role":role})

for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

message = st.chat_input("Send a message to the ai")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)