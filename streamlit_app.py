import streamlit as st
import chromadb
from chromadb import EphemeralClient
client = EphemeralClient()  # 메모리 전용
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    SimpleKeywordTableIndex,
    TreeIndex,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from IPython.display import Markdown, display
from clova_llama_index import ClovaClient, ClovaIndexEmbeddings, ClovaLLM

# --- UI 설정 ---
st.title("💬 공공조달 상담사")
st.write(
    """
    공공조달, 어렵게만 느껴지셨나요? \n
    이 챗봇은 중소기업, 벤처기업, 그리고 혁신기업 여러분이 조달 시장에 쉽게 다가갈 수 있도록 도와주는 든든한 가이드입니다. \n
    입찰 절차부터 지원 제도까지, 필요한 정보를 쉽고 빠르게 알려드릴게요.\n"""
)

# --- 클로바 API 설정 ---
client = ClovaClient(api_key="nv-c4042ebc0d004d3a8bca5b8a27c39261bDcZ")
llm = ClovaLLM(client)
embed_model = ClovaIndexEmbeddings(client, embed_batch_size=1)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

# --- 인덱싱 캐싱 함수 ---
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine

@st.cache_resource(show_spinner="테스트 문서를 인덱스 하는 중...")
def load_query_engine():
    documents = SimpleDirectoryReader("/workspaces/chatbot/testdata").load_data()
    nodes = Settings.node_parser.get_nodes_from_documents(documents)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    tree_index = TreeIndex(nodes, storage_context=storage_context)

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(response_mode="tree_summarize", use_async=True),
        name="vector",
        description="지정 데이터의 특정 부분을 검색할 때 유용"
    )
    keyword_tool = QueryEngineTool.from_defaults(
        query_engine=keyword_index.as_query_engine(response_mode="tree_summarize", use_async=True),
        name="keyword",
        description="키워드 기반 검색에 유용"
    )
    tree_tool = QueryEngineTool.from_defaults(
        query_engine=tree_index.as_query_engine(response_mode="tree_summarize", use_async=True),
        name="tree",
        description="전체 데이터 협성과 구조 이해에 유용"
    )

    obj_index = ObjectIndex.from_objects(
        [vector_tool, keyword_tool, tree_tool],
        index_cls=VectorStoreIndex,
    )
    return ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

# --- 쿼리 엔진 불러오기 ---
query_engine = load_query_engine()

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 이전 메시지 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("궁금한 내용을 입력해 주세요"):
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = query_engine.query(
        """
            아래 질문에 대해서 context를 충분히 고려해서
            항상 한국말로 대답해줘
            {prompt}
        """)

    with st.chat_message("assistant"):
        st.markdown(response.response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.response
        })
