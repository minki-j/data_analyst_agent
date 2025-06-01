from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
)
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
    get_save_variables_node,
    get_init_e2b_sandbox_node,
    get_python_executor_node,
    StepState,
    prompt_asking_agent_to_write_code_iteratively,
    retry_policy,
    samples_of_using_done,
)

CHECKLIST = """

""".strip()

CRITIC_GUIDE = """

""".strip()

MAX_MESSAGE_TURN = 50


def init_message_history(state: StepState):
    writer = get_stream_writer()
    writer({"oneline_message": "📈 Initiating step 4...", "current_step": 4})

    input_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """
You are a ReAct(reason and act) data analyst agent specializing in wiritng code to achieve the objective. 
You are currently in the fourth step of the data analysis process: Data Analysis. Your primary responsibility is to analyze the data to answer the objective.

Focus on:

- Statistical analysis and hypothesis testing
- Advanced data modeling and pattern analysis
- Creating comprehensive visualizations and charts
- Comparative analysis across different segments
- Trend analysis and forecasting (if applicable)
- Performance metrics calculation
- Business insights extraction
- Interactive dashboards and reporting
- Advanced statistical techniques (regression, clustering, etc.)
- Predictive analysis where appropriate

Your goal is to transform the exploratory insights into actionable analysis with compelling visualizations that answer the research questions and support decision-making. Generate publication-ready charts, graphs, and statistical summaries. 

---

## How to use "DONE" in your response
After the analysis and visualization work is complete, simply return "DONE" (all capital letters) as your response without any other text or code block.

{samples_of_using_done}

---

## How to write code
{prompt_asking_agent_to_write_code_iteratively}

--- 

## Additional context
The final objective of this whole data analysis is the following: {objective} This is where you should focus your analysis efforts to address this objective.

Here is the description and samples of the data that you have access to:
{variable_descriptions}

---

## Previous step summary
Before this step, you have already explored the data. Here is the summary of the exploration findings:
{exploration_summary}

Build upon these exploration insights to conduct deeper analysis and create visualizations!

---

Important Rules:
- You can write only one python code block in each response. Don't add more than one code block.
- Focus on analysis and visualization, not basic exploration.
- Make sure to write the code in the code block.
- Refrain from writing the entire code at once.
- Once analysis and visualization work is all done, simply return "DONE" (all capital letters) as your response without any other text or code block.
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
        prompt_asking_agent_to_write_code_iteratively=prompt_asking_agent_to_write_code_iteratively,
        samples_of_using_done=samples_of_using_done,
        objective=state.objective,
        variable_descriptions=get_variable_descriptions(state.variables),
        exploration_summary=next(
            step.report for step in state.steps if step.order == 3
        ),
    )

    return {
        "step_message_history": input_messages,
    }


def write_step_report(state: StepState):
    writer = get_stream_writer()
    writer({"oneline_message": "📋 Compiling analysis results..."})

    report = (reasoning_model_large | StrOutputParser()).invoke(
        [
            *state.step_message_history,
            HumanMessage(
                content="""
Great job! Now, summarize and write a short report about what you did in this step. This report will be passed to the further steps, and the message history will be cleared. This is where you can keep inportant information that you want to keep in mind for the next steps.

This step was the most important step in the data analysis process. If you miss important information here, then the final report will be incomplete.

Try to make sure that how your conclusion or result is derived from the data. When the executive reads the report, they should be able to understand what methods are used to derive the result. For example, if you created a scoring model, you should explain the details of the model such as what features are used, what weights are assigned to each feature, etc.

Don't say "DONE" or write any more code. Your task is finished and now you have to write a report about what you did in this step.
""".strip()
            ),
        ]
    )

    writer({"oneline_message": "✅ Data analysis completed!"})

    for step in state.steps:
        if step.order == 4:
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
    step_name="Step 4: Data Analysis",
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
