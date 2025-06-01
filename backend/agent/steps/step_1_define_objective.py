from pydantic import BaseModel, Field
from typing import Annotated, Optional

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.types import Command, interrupt
from langgraph.config import get_stream_writer

from agent.llms import chat_model_openai_first
from agent.state import OverallState
from agent.utils import get_variable_descriptions, get_dataframe_info
from agent.common import get_checklist_validator_node, ValidationResult, retry_policy

CHECKLIST = """
- Is there any ambiguous term used? 
- Did the user provide what they wants at the end of the analysis?
- Did the user provide high level method how to achieve the objective?
""".strip()

MAX_MESSAGE_TURN = 3


class StepState(OverallState):
    step_message_history: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )
    checklist_validation_result: Optional[ValidationResult]


def init_message_history(state: StepState):
    writer = get_stream_writer()
    writer({"oneline_message": "1️⃣ Initializing the step 1...", "current_step": 1})

    input_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """
You are a data analyst agent. Currently you are in the first step of the data analysis process: Define the objective. In this step your goal is to define the objective of this data analysis with the user. 

The user will submit their request for the data analysis. Examine the request and first check if the request is answerable with the provided data. If it isn't explain why and suggest new objectives. If the request is answerable, but it's not specific enough, ask the user for more details with some suggestions. 

Try not to exceed 3 rounds of questions to the user. For example, if the use provided you new objective three times already, then you might stop asking clarification unless the request is extremely vague or there seems like an error in the latest message. 

When checking whether the request is specific enough, use the following criteria:
{step_1_checklist}

Here is the variables that you have access to.
{variable_descriptions}
{dataframe_info}
    """.strip()
            ),
            HumanMessagePromptTemplate.from_template(
                """
User request: {user_request}
    """.strip()
            ),
        ]
    ).format_messages(
        step_1_checklist=CHECKLIST,
        variable_descriptions=get_variable_descriptions(state.variables),
        user_request=state.objective,
        dataframe_info=get_dataframe_info(state.variables),
    )

    return {
        "step_message_history": input_messages,
    }


def agent(state: StepState):
    writer = get_stream_writer()

    writer({"oneline_message": "🔍 Reviewing the user request..."})

    class Schema(BaseModel):
        chain_of_thought: str = Field(
            description="Use this field to think aloud to reason whether the user request is answerable and specific enough to proceed to the next step."
        )
        is_request_answerable: bool
        is_request_specific: bool
        message_to_user: str = Field(
            description="If either any of the above is false, you can use this field to send a message to the user to a) explain why the request is not answerable or specific enough and suggest possible new objectives or b) ask the user for more details with some suggestions and suggest more specific objectives. Keep this message short and concise."
        )

    response: Schema = chat_model_openai_first.with_structured_output(Schema).invoke(
        state.step_message_history
    )

    # If the request is answerable and specific, we can proceed to the validation nodes
    if len(state.step_message_history) > (2 * MAX_MESSAGE_TURN):
        writer(
            {
                "stream_message": f"⏰ Message turn exceeded the limit {MAX_MESSAGE_TURN}. Ending the step 1."
            }
        )
        # But if the message turn is too long, we just end the process
        return Command(goto=END)

    if response.is_request_answerable and response.is_request_specific:
        writer({"oneline_message": "✅ Checklist passed"})

    else:
        # If the request is not answerable or specific, we need to ask the user for more details
        user_response = interrupt(
            {
                "message_to_user": response.message_to_user,
            }
        )

        new_objective = (chat_model_openai_first | StrOutputParser()).invoke(
            [
                HumanMessage(
                    content=f"""
Based on the user's response, update the objective.

Current objective:
{state.objective}

---

Agent's message:
{response.message_to_user}

User's response:
{user_response}

---

Important:
- Don't say "okay, I'll update the objective" or "Updated objective: " or anything like that. Just return the updated objective.
- Keep the format of the current objective.
- Don't use xml tags in the response. Just return the content of the objective.
""".strip()
                )
            ]
        )

        return Command(
            update={
                "step_message_history": [
                    AIMessage(content=response.message_to_user),
                    HumanMessage(
                        content=f"""
The user said:
{user_response}

And we updated the objective to:
{new_objective}
""".strip()
                    ),
                ],
                "objective": new_objective,
            },
            goto=agent.__name__,
        )


def rendevous(state: StepState):
    if (
        state.checklist_validation_result is not None
        and state.checklist_validation_result.pass_the_validation
    ):
        return Command(goto=END)
    else:
        return Command(goto=agent.__name__)


g = StateGraph(OverallState)
g.add_edge(START, init_message_history.__name__)

g.add_node(init_message_history, retry=retry_policy)
g.add_edge(init_message_history.__name__, agent.__name__)

g.add_node(
    agent, destinations=(agent.__name__, "checklist_validator", END), retry=retry_policy
)

g.add_node(get_checklist_validator_node(CHECKLIST, StepState), retry=retry_policy)
g.add_edge("checklist_validator", rendevous.__name__)

g.add_node(rendevous, destinations=(agent.__name__, END), retry=retry_policy)

g = g.compile()
