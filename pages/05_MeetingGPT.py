import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from operator import itemgetter

# -----------------------------
# Streamlit 설정
# -----------------------------
st.set_page_config(page_title="Meeting GPT", page_icon="📼")
st.markdown("# Meeting GPT\nUpload a video and get a transcript, summary, and Q&A chatbot.")

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Transcript"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        llm=ChatOpenAI(temperature=0.1),
        max_token_limit=1000,
        return_messages=True,
    )

# -----------------------------
# 설정
# -----------------------------
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert assistant that answers questions based solely on the transcript of a meeting.
Do **not** make up information or rely on prior knowledge.
If the answer cannot be found in the transcript, respond with "I don't know."

------ Transcript ------
{context}
------------------------
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# -----------------------------
# 함수들
# -----------------------------
@st.cache_resource()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    files = sorted(glob.glob(f"{chunk_folder}/*.mp3"))
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            text_file.write(transcript.text)

@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    if not os.path.exists(audio_path):
        command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
        subprocess.run(command)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    os.makedirs(chunks_folder, exist_ok=True)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")

# -----------------------------
# 사이드바 업로드
# -----------------------------
with st.sidebar:
    video = st.file_uploader("Upload Video", type=["mp4", "avi", "mkv", "mov"])

# -----------------------------
# 메인 처리
# -----------------------------
if video:
    video_path = f"./.cache/{video.name}"
    audio_path = video_path.replace("mp4", "mp3")
    transcript_path = video_path.replace("mp4", "txt")
    chunks_folder = f"./.cache/chunks/{os.path.splitext(video.name)[0]}"
    with open(video_path, "wb") as f:
        f.write(video.read())

    with st.status("Processing..."):
        extract_audio_from_video(video_path)
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        transcribe_chunks(chunks_folder, transcript_path)

    # -----------------------------
    # 탭 구성
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Transcript", "Summary", "Q&A"])

    with tab1:
        st.session_state["active_tab"] = "Transcript"
        with open(transcript_path, "r") as f:
            st.write(f.read())

    with tab2:
        st.session_state["active_tab"] = "Summary"
        if st.button("Generate Summary"):
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)

            first_prompt = ChatPromptTemplate.from_template("""
            Write a concise summary for the following:\n{text}
            """)
            first_chain = first_prompt | ChatOpenAI(temperature=0.1) | StrOutputParser()
            summary = first_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template("""
            Refine the following summary using the context below if necessary.
            SUMMARY:\n{existing_summary}\nCONTEXT:\n{context}
            """)
            refine_chain = refine_prompt | ChatOpenAI(temperature=0.1) | StrOutputParser()

            for doc in docs[1:]:
                summary = refine_chain.invoke({
                    "existing_summary": summary,
                    "context": doc.page_content,
                })
            st.write(summary)

    with tab3:
        st.session_state["active_tab"] = "Q&A"
        retriever = embed_file(transcript_path)
        streaming_llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[])

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "history": RunnableLambda(
                    st.session_state["memory"].load_memory_variables
                ) | itemgetter("history"),
            }
            | chat_prompt
            | streaming_llm
        )

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])

# -----------------------------
# Q&A 탭에서만 입력 받기
# -----------------------------
if st.session_state["active_tab"] == "Q&A":
    query = st.chat_input("Ask me something about the meeting!")

    if query:
        st.session_state["messages"].append({"role": "user", "message": query})
        with st.chat_message("user"):
            st.markdown(query)

        result = chain.invoke(query)
        st.session_state["messages"].append({"role": "assistant", "message": result.content})
        with st.chat_message("assistant"):
            st.markdown(result.content)
