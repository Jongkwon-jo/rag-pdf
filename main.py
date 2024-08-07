import os
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

st.title("PDF 기반 GPT")
st.subheader("PDF 파일을 기준으로 무엇이든 물어보세요!")

# Set LangSmith environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

with st.sidebar:
    button = st.button("대화내용 초기화")
    uploaded_file = st.file_uploader("파일을 업로드 해주세요.", type=["pdf"])

    api_key = st.text_input("API KEY", type='password', value="api_key")
    # 프롬프트를 선택할 수 있는 옵션
    system_prompt = st.text_area("시스템 프롬프트",
                          "당신은 친절하게 답변하는 Assistant입니다. 간결하게 답변해 주세요.")
option = f"{system_prompt}" + "\n\n#Question: {question}"

# 메시지 기록을 위한 저장소를 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if button:
    # 대화 내용 초기화
    st.session_state["messages"] = []

if not os.path.exists(".cache"):
    os.mkdir(".cache")


retrieval_chain = None

# 파일 업로드하는 함수
def upload_file(file):
    file_content = file.read()
    file_path = f"./.cache/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Step 1: 문서 로드
    # PDF 파일 로드. 파일 경로 입력
    loader = PyPDFLoader(file_path)

    # Step 2: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    docs = loader.load_and_split(text_splitter)

    # Step 3: 벡터 저장소 생성 & 임베딩
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

    # Step 4: 검색기(retriever) -> 질문에 대한 유사도 검색
    retriever = vectorstore.as_retriever()

    return retriever


# 만약에 파일이 업로드가 된다면...
if uploaded_file:
    retriever = upload_file(uploaded_file)
    
    # Step 5: 프롬프트 작성, context: 검색기에서 가져온 문장, question: 질문
    template = """당신은 문서에 대한 정보를 바탕으로 답변하는 친절한 assistant 입니다. 무조건, 주어진 Context 바탕으로 답변해 주세요.
    답변에 대한 출저도 함께 제공해 주세요.

    #Context:
    {context}

    #Question
    {question}
    """
    prompt = ChatPromptTemplate.from_messages([("human", template)])

    # Step 6: OpenAI GPT 모델 설정
    model = ChatOpenAI(model="gpt-4o")

    # Step 7: 질문에 대한 답변을 찾기 위한 체인 생성
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

# 이전 대화기록 전체를 출력해주는 함수
def print_chat_message():
    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)

print_chat_message()

user_input = st.chat_input('질문을 입력하세요.')

# 메시지를 추가만 해주는 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

if user_input:
    st.chat_message('user').write(user_input)
    # ChatGPT에 질문
    with st.chat_message("assistant"):
        response_container = st.empty()
        if retrieval_chain is not None:
            chunk = []

            answer = retrieval_chain.stream(user_input)
            tmp_answer = '<PDF 기반 답변을 하는 챗봇입니다>\n\n'
            for c in answer:
                tmp_answer += c
                response_container.markdown(tmp_answer)
                chunk.append(c)
            
            response = "".join(chunk)
        else:
            # 체인을 생성
            prompt = PromptTemplate.from_template(option)
            model = ChatOpenAI(model="gpt-4o", temperature=0)

            # 체인을 생성
            chain = prompt | model | StrOutputParser()# 체인을 생성
            chunk = []

            answer = chain.stream({"question": user_input})
            tmp_answer = '<일반 답변을 하는 챗봇입니다>\n\n'
            for c in answer:
                tmp_answer += c
                response_container.markdown(tmp_answer)
                chunk.append(c)
            
            response = "".join(chunk)
    # 메시지 기록을 위해 메시지를 캐싱에 저장합니다.
    add_message("user", user_input)
    add_message("assistant", response)
