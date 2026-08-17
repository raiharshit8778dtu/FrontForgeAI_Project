from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from typing import TypedDict, List, Any , Dict, cast
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage


# -----------------
# 1.LLM
# -----------------

model = ChatOpenRouter(model='gpt-4o-mini', temperature=0.7)

# Creating class for structured output 
class Ambiguity_Model(BaseModel):
    ambiguity_status: bool = Field(description='Check whether the question is ambiguous',default=True)
    clarification_question: str = Field(description="Generate a clarification question if the user's prompt is ambiguous")

llm = model.with_structured_output(Ambiguity_Model)

# ---------------------
# 2.1. ClarificationState
# ---------------------
class ClarificationState(TypedDict):
    """
    This state is used to handle clarification questions and answers.
    """
    user_prompt: str
    ambiguity_status: bool
    clarification_ques: str
    clarification_answer: str
    unambiguous_prompt: str

def ambiguity_dectector(state: ClarificationState) -> ClarificationState:
    """This function detects ambiguity in the user's input and generates a clarification question if needed."""

    user_prompt = state['user_prompt']
    response = llm.invoke([
        SystemMessage(content="You are an AI assistant that detects ambiguity in user input and generates clarification questions."),
        HumanMessage(content=f"User input: {user_prompt}\n\nPlease determine if the input is ambiguous. If it is, generate a clarification question. If not, indicate that the input is clear.")
    ])

    # With structured output, the result is the Pydantic model itself, not a message wrapper.
    if isinstance(response, Ambiguity_Model):
        state['ambiguity_status'] = response.ambiguity_status
        state['clarification_ques'] = response.clarification_question if response.ambiguity_status else ""
    else:
        state['ambiguity_status'] = False
        state['clarification_ques'] = ""
    return state

def get_response(state:ClarificationState) -> ClarificationState:
    clarification_question = state['clarification_ques']
    if clarification_question:
        # If there's a clarification question, we need to get the user's answer.
        user_answer = input(f"Clarification needed: {clarification_question}\nYour answer: ")
        state['clarification_answer'] = user_answer
        # I want to loop the process until the user prompt is completely unambiguous. For now, let's just append the clarification answer to the original prompt.
        state['user_prompt'] = f"{state['user_prompt']} (clarified with: {user_answer})"
    else:
        # If there's no clarification question, the unambiguous prompt is just the original user prompt.
        state['unambiguous_prompt'] = state['user_prompt']
    return state

def ambiguity_evaluation(state: ClarificationState) -> str:
    """Return the next node to execute based on whether the prompt is ambiguous."""
    if state['ambiguity_status'] is False:
        return END  # Exit clarification subgraph when clear
    return 'ambiguity_detector'  # Loop back to ambiguity detection if still ambiguous

# ------------------------
# 2.2. Clarification Graph (Subgraph 1)
# ------------------------
clarification_builder = StateGraph(ClarificationState)
clarification_builder.add_node('ambiguity_detector', ambiguity_dectector)
clarification_builder.add_node('get_response', get_response)

clarification_builder.add_edge(START, 'ambiguity_detector')
clarification_builder.add_edge('ambiguity_detector', 'get_response')
clarification_builder.add_conditional_edges('get_response', ambiguity_evaluation)

clarification_subgraph = clarification_builder.compile()


# -------------------
# 3.1. Planner State & Planner Subgraph (Subgraph 2)
# -------------------
class PlannerState(TypedDict):
    unambiguous_prompt: str
    plan_of_action: str

def planner(state: PlannerState) -> PlannerState:
    instructions = state.get('unambiguous_prompt', '')
    response = model.invoke([
        SystemMessage(content='You are an efficient planner agent who decomposes the user requirements into a structured plan; listing pages, components and dependencies.'),
        HumanMessage(content=f'User instructions:\n{instructions}\nList down a structured plan to create the frontend application as per the user requirements.')
    ])
    state['plan_of_action'] = str(response.content)
    return state

planner_builder = StateGraph(PlannerState)
planner_builder.add_node('planner', planner)
planner_builder.add_edge(START, 'planner')
planner_builder.add_edge('planner', END)

planner_subgraph = planner_builder.compile()


# ------------------------
# 4. Parent Orchestrator Graph (OverallState)
# ------------------------
class OverallState(TypedDict):
    user_prompt: str
    unambiguous_prompt: str
    plan_of_action: str

checkpointer = InMemorySaver()
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

parent_builder = StateGraph(OverallState)

# Register subgraphs as nodes in the parent orchestrator
parent_builder.add_node('clarification_agent', clarification_subgraph)
parent_builder.add_node('planner_agent', planner_subgraph)

# Define parent pipeline flow
parent_builder.add_edge(START, 'clarification_agent')
parent_builder.add_edge('clarification_agent', 'planner_agent')
parent_builder.add_edge('planner_agent', END)

workflow = parent_builder.compile(checkpointer=checkpointer)

# ------------------------
# 5. Execution Test
# ------------------------
initial_state: OverallState = {
    'user_prompt': 'Code for a personal portfolio website with projects and contact form',
    'unambiguous_prompt': '',
    'plan_of_action': ''
}

if __name__ == '__main__':
    final_state = workflow.invoke(initial_state, config=CONFIG)   # type: ignore
    print("\n================ FINAL PLAN OF ACTION ================\n")
    print(final_state.get('plan_of_action'))