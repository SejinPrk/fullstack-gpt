from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage, HumanMessage
import streamlit as st

# 세션 변수 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )


# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# LLM 응답을 스트리밍 방식으로 표시하기 위한 콜백 핸들러 클래스
class ChatCallbackHandler(BaseCallbackHandler):
    message = "" # 빈 메세지로 시작

    # self는 class 전체를 참조하므로 사용 (객체지향 문법)
    # LLM이 새 토큰 생성 -> empty box 생성됨
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty() # 동적으로 업데이트할 빈 컨테이너 생성

    # LLM이 생성 완료 -> 최종 메시지 저장
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai") # 메시지 생성이 완료되면 세션에 저장

    # LLM이 새 토큰을 생성할 때마다 호출 -> 실시간 업데이트
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token # 토큰을 메세지에 추가하는 방식
        self.message_box.markdown(self.message) # UI 업데이트

# OpenAI 모델 설정 - 스트리밍 활성화 및 콜백 핸들러 연결
llm = ChatOpenAI(
    temperature=0.1, # 낮은 temperature로 일관된 응답 생성
    streaming=True, # 토큰 스트리밍 활성화
    callbacks = [
        ChatCallbackHandler(), # 스트리밍 처리를 위한 콜백 핸들러 연결
    ]
)

# 파일 임베딩 함수 - 캐싱 적용으로 성능 최적화 (파일이 달라질 경우에만 실행됨)
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # 파일 콘텐츠 읽기
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    # 파일 로컬에 저장
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 임베딩 캐시 디렉토리 설정
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # 텍스트 분할 설정 - 청크 크기와 오버랩 조정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600, # 각 청크의 최대 크기
        chunk_overlap=100, # 문맥 유지를 위한 청크 간 겹침 정도
    )

    # 파일 로드 및 분할
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # OpenAI 임베딩 모델 설정 및 캐싱
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # 벡터 저장소 생성 및 검색기 반환
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# 대화 기록을 포맷팅하는 함수
def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"AI: {message.content}\n"
    return formatted_history

# 메시지를 세션 상태에 저장하는 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

    # 메모리에도 메시지 추가
    if role == "human":
        st.session_state["memory"].chat_memory.add_message(HumanMessage(content=message))
    elif role == "ai":
        st.session_state["memory"].chat_memory.add_message(AIMessage(content=message))

# 메시지를 UI에 표시하고 필요시 저장하는 함수
def send_message(message, role, save=True):
    with st.chat_message(role): # Streamlit 채팅 메시지 UI 컴포넌트
        st.markdown(message)
    if save:
        save_message(message, role)

# 채팅 기록을 UI에 표시하는 함수
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False, # 이미 저장된 메시지이므로 다시 저장하지 않음
        )

# 문서에서 개행 문자로 구분된 것들을 하나의 string으로 합친다.
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# LLM에 전달할 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context and chat history.
            If you don't know the answer just say you don't know. Don't make anything up.

            Context: {context}

            Chat History:
            {chat_history}
            """
        ),
        ("human", "{question}")
    ]
)

# 앱 제목 설정
st.title("DocumentGPT")

# 앱 소개 메시지
st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
    """
)

# 사이드바에 파일 업로더 배치
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"], # 지원하는 파일 형식 지정
    )

# 파일이 업로드된 경우
if file:
    # 파일 임베딩 및 검색기 준비
    retriever = embed_file(file)

    # 시작 메시지 표시 (저장x)
    send_message("I'm ready! Ask away!", "ai", save=False)

    # 이전 대화 기록 표시
    paint_history()

    # 사용자 입력 필드 표시
    message = st.chat_input("Ask anything about your file...")

    # 사용자가 메시지를 입력한 경우
    if message:
        # 사용자 메시지 표시 및 저장
        send_message(message, "human")

        # RAG 파이프라인 구성:
        # 1. 컨텍스트와 질문 준비
        # 2. 프롬프트 템플릿 적용
        # 3. LLM으로 응답 생성
        chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),  # 문서 검색 및 포맷팅
                    "question": RunnablePassthrough(),  # 원본 질문 그대로 전달
                    "chat_history": RunnableLambda(
                        lambda _: format_chat_history(
                            st.session_state.get("memory", ConversationBufferMemory(
                                return_messages=True,
                                memory_key="chat_history"
                            )).chat_memory.messages
                        )
                    )
                }
                | prompt  # 프롬프트 템플릿 적용
                | llm  # LLM으로 응답 생성
        )

        # AI 응답 메시지 UI 컴포넌트 생성 및 체인 실행
        with st.chat_message("ai"):
            chain.invoke(message) # 메시지를 입력으로 체인 실행
else:
    # 파일이 없는 경우 메시지 및 메모리 초기화
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
