from langgraph.grraph import StateGraph,MessagesState,START,END

def mock_llm(state: MessageState):
    return {
        "messages": [{"role": "assistant", 
        "content": "Hello, how can I help you today?"}]
        }

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, mock_llm)
graph.add_edge(mock_llm, END)

graph.invoke({"messages": [{"role": "user", "content": "Hello"}]})