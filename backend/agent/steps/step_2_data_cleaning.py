from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from langgraph.config import get_stream_writer

from agent.llms import reasoning_model_large
from agent.state import OverallState
from agent.utils import get_variable_descriptions
from agent.common import (
    get_checklist_validator_node,
    get_critic_validator_node,
    get_rendevous_node,
    get_code_agent_node,
    get_init_e2b_sandbox_node,
    get_save_variables_node,
    get_python_executor_node,
    StepState,
    prompt_asking_agent_to_write_code_iteratively,
    retry_policy,
    samples_of_using_done,
)

CHECKLIST = """
- Address missing values
- Address duplicate records
- Address inconsistent formatting
- Address inconsistent naming conventions
- Address outliers
- Address data type mismatches
""".strip()

CRITIC_GUIDE = """

""".strip()

MAX_MESSAGE_TURN = 30


def init_message_history(state: StepState):
    writer = get_stream_writer()
    writer({"oneline_message": "🧹 Initializing step 2...", "current_step": 2})

    input_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """
You are a ReAct(reason and act) data analyst agent specializing in wiritng code to achieve the objective. 
You are currently in the second step of the data analysis process: Data Cleaning. Your primary responsibility is to prepare the data for exploration by identifying and addressing data quality issues such as:

{checklist}

---

## Scope of the task
Focus solely on cleaning and preprocessing the data. Do not perform any analysis or draw conclusions. Your goal is to ensure the data is in a clean, consistent format that will facilitate effective exploration in the next step. 

---

## How to use "DONE" in your response
After cleaning is done simply return "DONE" (all capital letters) as your response without any other text or code block. Then the processed dataset will be passed to the data exploration phase. 

{samples_of_using_done}

---

## How to write code
{prompt_asking_agent_to_write_code_iteratively}

---

## Additional context
The final objective of this whole data analysis is the following: {objective} This is just a reference. You don't need to achieve this objective in this step. Again, you are in the data cleaning step.

Here is the description and samples of the data that you have access to:
{variable_descriptions}

---

## Important Rules
- You can write only one python code block in each response. Don't add more than one code block.
- Make sure to write the code in the code block.
- Refrain from writing the entire code at once.
- Once cleaning is all done, simply return "DONE" (all capital letters) as your response without any other text or code block.
- Don't plot graphs. You are not a multi-modal agent. You can only understand text.
- Don't plot graphs. You are not a multi-modal agent. You can only understand text.
    """.strip()
            ),
            HumanMessagePromptTemplate.from_template(
                """
Okay, let's start.
    """.strip()
            ),
        ]
    ).format_messages(
        checklist=CHECKLIST,
        prompt_asking_agent_to_write_code_iteratively=prompt_asking_agent_to_write_code_iteratively,
        samples_of_using_done=samples_of_using_done,
        objective=state.objective,
        variable_descriptions=get_variable_descriptions(state.variables),
    )

    return {
        "step_message_history": input_messages,
    }


def write_step_report(state: StepState):
    writer = get_stream_writer()
    writer({"oneline_message": "📝 Writing data cleaning summary report..."})

    report = (reasoning_model_large | StrOutputParser()).invoke(
        [
            *state.step_message_history,
            HumanMessage(
                content="""
Great job! Now, summarize and write a short report about what you did in this step. This report will be passed to the further steps, and the message history will be cleared. This is where you can keep inportant information that you want to keep in mind for the next steps.

Don't say "DONE" or write any more code. Your task is finished and now you have to write a report about what you did in this step.
""".strip()
            ),
        ]
    )

    writer({"oneline_message": "✅ Data cleaning step completed!"})

    for step in state.steps:
        if step.order == 2:
            step.completed = True
            step.report = report
            break

    return Command(
        update={
            "steps": state.steps,
        },
        goto=END,
    )


# Shared nodes
save_variables = get_save_variables_node(StepState)
init_e2b_sandbox = get_init_e2b_sandbox_node(StepState)
python_executor = get_python_executor_node(StepState)
checklist_validator = get_checklist_validator_node(CHECKLIST, StepState)
critic_validator = get_critic_validator_node(CRITIC_GUIDE, StepState)
agent = get_code_agent_node(StepState, MAX_MESSAGE_TURN, save_variables.__name__)
rendevous = get_rendevous_node(
    next_node_name=save_variables.__name__,
    agent_node_name=agent.__name__,
    step_name="Step 2: Data Cleaning",
    state_type=StepState,
)


g = StateGraph(OverallState)
g.add_edge(START, init_e2b_sandbox.__name__)

g.add_node(init_e2b_sandbox)
g.add_edge(init_e2b_sandbox.__name__, init_message_history.__name__)

g.add_node(init_message_history)
g.add_edge(init_message_history.__name__, agent.__name__)

g.add_node(
    agent,
    destinations=(
        agent.__name__,
        python_executor.__name__,
        checklist_validator.__name__,
        critic_validator.__name__,
    ),
    retry=retry_policy,
)

g.add_node(python_executor, retry=retry_policy)
g.add_edge(python_executor.__name__, agent.__name__)

g.add_node(checklist_validator, retry=retry_policy)
g.add_edge(checklist_validator.__name__, rendevous.__name__)

g.add_node(critic_validator, retry=retry_policy)
g.add_edge(critic_validator.__name__, rendevous.__name__)

g.add_node(
    rendevous,
    destinations=(agent.__name__, save_variables.__name__),
    retry=retry_policy,
)

g.add_node(save_variables, retry=retry_policy)
g.add_edge(save_variables.__name__, write_step_report.__name__)

g.add_node(write_step_report, retry=retry_policy)
g.add_edge(write_step_report.__name__, END)

g = g.compile()

with open("step_2_data_cleaning.png", "wb") as f:
    f.write(g.get_graph().draw_mermaid_png())
