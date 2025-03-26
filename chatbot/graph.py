from langchain_google_vertexai import ChatVertexAI
from langchain_google_community import VertexAISearchRetriever
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import vertexai
from vertexai.preview.generative_models import SafetySetting
import time
from typing import Dict, Any

# langchain.debug = True
safety_settings = {
    SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
}

PROJECT_ID = "dev-range-451608-c1"
LOCATION = "europe-west9"
DATA_STORE_ID = "demo_1740128963679"
LOCATION_ID = "eu"
MODEL = "gemini-2.0-flash-001"
temperature = 0
top_p = 0
vertexai.init(project=PROJECT_ID, location=LOCATION)
llm = ChatVertexAI(
    model_name=MODEL,
    temperature=temperature,
    top_p=top_p,
    safety_settings=safety_settings,
)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    try:
        # 添加缓存检查
        cache_key = hash(query)
        if hasattr(retrieve, "_cache") and cache_key in retrieve._cache:
            return retrieve._cache[cache_key]

        retriever = VertexAISearchRetriever(
            project_id=PROJECT_ID,
            data_store_id=DATA_STORE_ID,
            location_id=LOCATION_ID,
            engine_data_type=1,
            max_documents=10,
        )

        retrieved_docs = retriever.invoke(query)

        # 处理空结果情况
        if not retrieved_docs:
            return "No relevant information found.", []

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )

        # 存入缓存
        if not hasattr(retrieve, "_cache"):
            retrieve._cache = {}
        retrieve._cache[cache_key] = (serialized, retrieved_docs)

        return serialized, retrieved_docs
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        return f"An error occurred while retrieving information: {str(e)}", []


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    try:
        # 获取工具消息，限制历史消息数量
        MAX_HISTORY = 5
        recent_tool_messages = []
        for message in reversed(state["messages"][-MAX_HISTORY:]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # 格式化提示信息
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        # 优化系统消息，添加更多控制参数
        system_message_content = (
            """You are a highly intelligent and empathetic conversational assistant designed to simulate interactions with individuals who represent specific hair archetypes. 
        Your primary role is to help cosmetic industry Insights teams better understand their preferences, routines, and needs.
        Your knowledge is enriched by a RAG system, which provides detailed, objective information about each hair archetype. 
        
        Response Guidelines:
        1. Keep responses concise and focused (max 20 sentences)
        2. Prioritize recent context and user's immediate needs
        3. Include specific, actionable recommendations
        4. Use natural, conversational language
        5. Reference retrieved information when available
        
        Archetypes:
        - POLISHED OPULENCE
        - 3D CURLS
        - CELEBRATE MY COILS
        - LUMINOUS CHIC
        - AIRY COOL SLIM FACE
        - NATURAL SUBLIMATION
        - YOUTH KEEPER
        
        Key Focus Areas:
        - Hair care preferences and routines
        - Product recommendations
        - Styling techniques
        - Hair health concerns
        - Personal care habits"""
            "\n\n"
            f"Retrieved Context:\n{docs_content}"
        )

        # 过滤并限制对话历史
        conversation_messages = [
            message
            for message in state["messages"][-MAX_HISTORY:]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]

        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # 生成响应
        response = llm.invoke(prompt)
        return {"messages": [response]}
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        return {"messages": [AIMessage(content=error_message)]}


# Build graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge(START, "generate")

graph_builder.set_entry_point("generate")
graph_builder.add_edge("generate", END)
# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# def monitor_performance(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         duration = end_time - start_time
#         print(f"{func.__name__} took {duration:.2f} seconds to execute")
#         return result

#     return wrapper


# @monitor_performance
# def is_there_tool_calls(state: MessagesState):
#     last_message = state["messages"][-1]
#     if last_message.tool_calls:
#         return "tool_node"
#     else:
#         return "__end__"


# class GraphManager:
#     def __init__(self):
#         self.graph_builder = StateGraph(MessagesState)
#         self.tool_node = ToolNode([retrieve])
#         self.setup_graph()

#     def setup_graph(self):
#         self.graph_builder.add_node("agent", generate)
#         self.graph_builder.add_node("tool_node", self.tool_node)
#         self.graph_builder.add_conditional_edges("agent", is_there_tool_calls)
#         self.graph_builder.add_edge("tool_node", "agent")
#         self.graph_builder.set_entry_point("agent")

#     def get_config(self, thread_id: str = None) -> Dict[str, Any]:
#         return {
#             "configurable": {
#                 "thread_id": thread_id or "default_thread",
#                 "max_iterations": 10,
#                 "timeout": 30,
#             }
#         }

#     @monitor_performance
#     def invoke_graph(self, messages: str, thread_id: str = None) -> str:
#         try:
#             graph = self.graph_builder.compile()
#             final_state = graph.invoke(
#                 {"messages": [{"role": "user", "content": messages}]},
#                 config=self.get_config(thread_id),
#             )
#             return final_state["messages"][-1].content
#         except Exception as e:
#             error_message = f"Error in graph execution: {str(e)}"
#             print(error_message)
#             return error_message


# # 创建全局图管理器实例
# graph_manager = GraphManager()


# def invoke_our_graph(st_messages: str) -> str:
#     """Wrapper function for backward compatibility"""
#     return graph_manager.invoke_graph(st_messages)
