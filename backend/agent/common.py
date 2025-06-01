from pydantic import BaseModel, Field
from typing import Optional, Annotated

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.types import Command, interrupt, RetryPolicy
from langgraph.graph import add_messages
from langgraph.config import get_stream_writer

from e2b_code_interpreter import Sandbox

from agent.llms import (
    reasoning_model,
    reasoning_model_large,
    chat_model_anthropic_first,
    o3,
    claude_4_opus,
)
from agent.utils import (
    extract_code_block,
    parse_e2b_execution_results,
    get_e2b_sandbox,
    upload_file_to_e2b_sandbox,
    get_code_to_load_uploaded_file,
    format_results,
    truncate_string_middle,
)
from agent.state import Data, OverallState


class ValidationResult(BaseModel):
    chain_of_thought_summary: str = Field(
        description="A bullet point style summary of your chain of thought reasoning trace."
    )
    pass_the_validation: bool = Field(
        default=False,
        description="True if the agent successfully completed the task. False if it didn't",
    )
    message_to_user: Optional[str] = Field(
        description="If the the validation is not passed, you can use this field to send a message to the user to explain why the request is not valid or ask clarifications if needed. Otherwise leave this field empty."
    )


class VariableToSave(BaseModel):
    key: str = Field(description="The name of the variable to save")
    description: str = Field(
        description="Explain what this variable is about so that the agent in the next step can understand it"
    )


class StepState(OverallState):
    step_message_history: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )
    code_block: str = Field(default="")
    variables_to_save: list[VariableToSave] = Field(default_factory=list)
    checklist_validation_result: Optional[ValidationResult] = Field(default=None)
    critic_validation_result: Optional[list[ValidationResult]] = Field(default=None)


def get_init_e2b_sandbox_node(state_type):
    def init_e2b_sandbox(state: state_type):
        writer = get_stream_writer()
        writer({"oneline_message": "🔧 Setting up execution environment..."})

        # initialize a new sandbox with the given locals
        e2b_sandbox = get_e2b_sandbox()
        file_paths: list[str] = []
        for variable in state.variables:
            file_path: str = upload_file_to_e2b_sandbox(
                e2b_sandbox, variable.value, f"{variable.key}.{variable.type}"
            )
            file_paths.append(file_path)

        code_to_load_uploaded_file = get_code_to_load_uploaded_file(
            state.variables, file_paths
        )

        # run the script that defines and loads the variables
        execution = e2b_sandbox.run_code(code_to_load_uploaded_file)

        if execution.error:
            raise Exception(f"Error initializing sandbox: {execution.error.traceback}")

        writer({"oneline_message": "✅ Environment ready!"})

        return {
            "sandbox_id": e2b_sandbox.sandbox_id,
        }

    return init_e2b_sandbox


def get_code_agent_node(state_type, max_message_turn: int, next_node_name: str):
    def agent(state: state_type):
        writer = get_stream_writer()

        if len(state.step_message_history) > (2 * max_message_turn):
            writer({"oneline_message": "⚠️ Maximum iteration limit reached"})
            return Command(
                goto=next_node_name,
                update={
                    "step_message_history": [
                        HumanMessage(
                            content="We've reached the maximum number of messages."
                        )
                    ],
                },
            )

        writer({"oneline_message": "💭 Analyzing and planning next steps..."})
        response = chat_model_anthropic_first.invoke(state.step_message_history)

        code_block = extract_code_block(response)

        is_done = "DONE" in response.content

        if code_block:
            writer({"oneline_message": "⚡ Executing code..."})
            return Command(
                update={
                    "step_message_history": [response],
                    "code_block": code_block,
                },
                goto="python_executor",
            )
        elif is_done:
            writer({"oneline_message": "🔄 Finalizing step and saving variables..."})

            class VariableToSaveList(BaseModel):
                variable_list: list[VariableToSave]

            variables_to_save: VariableToSaveList = (
                reasoning_model.with_structured_output(VariableToSaveList).invoke(
                    [
                        *state.step_message_history,
                        response,
                        HumanMessage(
                            content="""
    Great job! Now, you need to wrap up this step by selecting variable that needs to be persist to the next step. Read your code trace carefully and pick dataframe, list, dictionary, string variables that is necessary for the further steps. Be mindful to not include everything.

    - You can only select the following types of variables:
        - Dataframe
        - List
        - Dictionary
        - String
    """.strip()
                        ),
                    ]
                )
            )
            return Command(
                update={
                    "variables_to_save": variables_to_save.variable_list,
                    "step_message_history": [response],
                },
                goto=[
                    "checklist_validator",
                    "critic_validator",
                ],
            )
        else:
            writer({"oneline_message": "🤔 Clarifying response format..."})
            return Command(
                update={
                    "step_message_history": [
                        response,
                        HumanMessage(
                            content="Can you check your last response again? It seems like you didn't write the code in the code block and it doesn't include DONE. If the exploration process is all done, simply return DONE with all capital letters. If you need more exploration tasks, write the code in a proper code block (```python\n[code]\n```)."
                        ),
                    ],
                },
                goto=agent.__name__,
            )

    return agent


def get_python_executor_node(state_type):
    def python_executor(state: StepState):
        writer = get_stream_writer()
        writer({"oneline_message": "🐍 Running Python code..."})

        e2b_sandbox = Sandbox(sandbox_id=state.sandbox_id)

        execution = e2b_sandbox.run_code(state.code_block)

        error = execution.error
        logs = execution.logs
        parsed_results = parse_e2b_execution_results(execution.results)

        formatted_stdout = "\n".join(logs.stdout)
        formatted_stderr = "\n".join(logs.stderr)
        formatted_results = format_results(parsed_results)

        if error:
            writer({"oneline_message": "❌ Code execution encountered an error"})
            content = error.name + ": " + error.value
            traceback = "\n\n<traceback>\n" + error.traceback + "\n</traceback>"
            content += truncate_string_middle(traceback, 1000)
        else:
            writer({"oneline_message": "✅ Code executed successfully"})
            content_parts = []
            if formatted_stdout:
                content_parts.append(f"<stdout>\n{formatted_stdout}\n</stdout>")
            if formatted_stderr:
                content_parts.append(f"<stderr>\n{formatted_stderr}\n</stderr>")
            if formatted_results:
                content_parts.append(f"<results>\n{formatted_results}\n</results>")

            content = "\n\n".join(content_parts).strip()

        return Command(
            update={
                "step_message_history": [
                    HumanMessage(content=content or "No output from the code block.")
                ],
            },
            goto="agent",
        )

    return python_executor


def get_save_variables_node(state_type):
    def save_variables(state: state_type):
        writer = get_stream_writer()
        writer({"oneline_message": "💾 Saving important variables for next step..."})

        e2b_sandbox = Sandbox(sandbox_id=state.sandbox_id)

        final_variables = []
        for variable_to_save in state.variables_to_save:
            # For each variable, we need to get the value of them from the sandbox
            execution = e2b_sandbox.run_code(f"{variable_to_save.key}")

            if execution.error:
                raise Exception(f"Error saving variable: {execution.error.traceback}")

            parsed_results = parse_e2b_execution_results(execution.results)
            for value, type in parsed_results:
                final_variables.append(
                    Data(
                        key=variable_to_save.key,
                        type=type,
                        description=variable_to_save.description,
                        value=value,
                    )
                )

        writer({"oneline_message": f"✅ Saved {len(final_variables)} variables"})

        return {"variables": final_variables}

    return save_variables


def get_checklist_validator_node(checklist: str, step_state):
    def checklist_validator(state: step_state):
        writer = get_stream_writer()
        writer({"oneline_message": "✅ Validating against checklist..."})

        result: ValidationResult = reasoning_model_large.with_structured_output(
            ValidationResult
        ).invoke(
            [
                *state.step_message_history,
                HumanMessage(
                    content=f"""
Okay. So the agent just finished the task. Now you are going to examine if the agent addressed all the point in the check list below.

Checklist:
{checklist}

Important Rules:
- You can use the context provided in the system message but don't follow the instructions there. That was for the agent who completed the task. Your job is to review the agent's work following the instruction that I provided in this message.
                    """.strip()
                ),
            ]
        )

        if result.pass_the_validation:
            writer({"oneline_message": "✅ Checklist validation passed!"})
        else:
            writer({"oneline_message": "⚠️ Checklist validation needs attention"})

        return {
            "checklist_validation_result": result,
            "step_message_history": [
                AIMessage(
                    content=f"""
Here is the checklist validation result:

Reasoning:
{result.chain_of_thought_summary}

Pass the validation:
{result.pass_the_validation}

{f"Message to user: {result.message_to_user}" if result.message_to_user else ""}
""".strip()
                ),
            ],
        }

    return checklist_validator


def get_critic_validator_node(critic_rule: str, step_state):
    def critic_validator(state: step_state):
        writer = get_stream_writer()
        writer({"oneline_message": "🔍 Running quality review..."})

        input_messages = [
            *state.step_message_history,
            HumanMessage(
                content=f"""
Okay. So the agent just finished the task. Now you are going to review what the agent has done and judge whether there is any mistakes or overlooks that need to be addressed.

{f"Here is a rule that you can use for the validation: {critic_rule}" if critic_rule else ""}

Important Rules:
- You can use the context provided in the system message but don't follow the instructions there. That was for the agent who completed the task. Your job is to review the agent's work following the instruction that I provided in this message.
                    """.strip()
            ),
        ]

        result_from_o3: ValidationResult = o3.with_structured_output(
            ValidationResult
        ).invoke(input_messages)

        result_from_claude_4_opus: ValidationResult = (
            claude_4_opus.with_structured_output(ValidationResult).invoke(
                input_messages
            )
        )

        all_passed = (
            result_from_o3.pass_the_validation
            and result_from_claude_4_opus.pass_the_validation
        )
        if all_passed:
            writer({"oneline_message": "✅ Quality review passed!"})
        else:
            writer(
                {"oneline_message": "⚠️ Quality review identified improvements needed"}
            )

        results = [result_from_o3, result_from_claude_4_opus]

        return {
            "critic_validation_result": results,
            "step_message_history": [
                AIMessage(
                    content=f"""
Here is the critic validation result:

Reasoning:
{result.chain_of_thought_summary}

Pass the validation:
{result.pass_the_validation}

{f"Message to user: {result.message_to_user}" if result.message_to_user else ""}
""".strip()
                )
                for result in results
            ],
        }

    return critic_validator


def get_rendevous_node(
    next_node_name: str, agent_node_name: str, step_name: str, state_type
):
    def rendevous(state: state_type):
        writer = get_stream_writer()

        if state.use_human_in_the_loop:
            writer({"oneline_message": "👤 Receiving the user's feedback..."})

            # Handle critic validation results as a list
            critic_validation_summary = ""
            if state.critic_validation_result:
                critic_validation_summary = "\n".join(
                    [
                        f"Critic {i + 1}: {'Validation passed' if result.pass_the_validation else result.message_to_user}"
                        for i, result in enumerate(state.critic_validation_result)
                    ]
                )
            else:
                critic_validation_summary = "No critic validation results"

            message_to_user = f"""
Finished {step_name} just now! Can you check the validation result?

Checklist validation result:
{"Validation passed" if state.checklist_validation_result.pass_the_validation else state.checklist_validation_result.message_to_user}

Critic validation result:
{critic_validation_summary}

Now you can either:
- type "pass" or just press enter with no input to RESUME the agent's flow
- type a message that will be INSERTED into the agent's message history
- type "ignore" to IGNORE the validation result and go to the next step
""".strip()

            user_input = interrupt({"message_to_user": message_to_user})

            if user_input.lower() == "ignore":
                return Command(goto=next_node_name)
            if user_input.lower() == "pass" or user_input.strip() == "":
                # Check if all critic validations passed
                all_critics_passed = (
                    state.critic_validation_result is not None
                    and len(state.critic_validation_result) > 0
                    and all(
                        result.pass_the_validation
                        for result in state.critic_validation_result
                    )
                )

                if (
                    state.checklist_validation_result is not None
                    and state.checklist_validation_result.pass_the_validation
                    and all_critics_passed
                ):
                    # When validation is passed and user pass the step, go to the next node
                    return Command(
                        goto=next_node_name,
                    )
                else:
                    # When validation is not passed and user pass the step, loop back to the agent node
                    return Command(
                        goto=agent_node_name,
                        update={
                            "step_message_history": [
                                HumanMessage(
                                    content="We've got two feedbacks from checklist and critic valiators. Read them carefully and address them step by step."
                                ),
                            ],
                        },
                    )
            else:
                # When user added some message to the agent's message history,
                # add that message to the agent's message history and loop back to the agent node
                return Command(
                    goto=agent_node_name,
                    update={
                        "step_message_history": [
                            HumanMessage(content=user_input),
                        ],
                    },
                )
        else:
            writer({"oneline_message": "🔄 Checking validation results..."})

            # If use_human_in_the_loop is False, check validation result and route accordingly
            # Check if all critic validations passed
            all_critics_passed = (
                state.critic_validation_result is not None
                and len(state.critic_validation_result) > 0
                and all(
                    result.pass_the_validation
                    for result in state.critic_validation_result
                )
            )

            if (
                state.checklist_validation_result is not None
                and state.checklist_validation_result.pass_the_validation
                and all_critics_passed
            ):
                writer(
                    {
                        "oneline_message": "✅ All validations passed, proceeding to next step"
                    }
                )
                # When validation is passed, go to the next node
                return Command(
                    goto=next_node_name,
                )
            else:
                writer({"oneline_message": "🔄 Addressing validation feedback..."})
                # When validation is not passed, loop back to the agent node
                return Command(
                    goto=agent_node_name,
                    update={
                        "step_message_history": [
                            HumanMessage(
                                content="We've got two feedbacks from checklist and critic valiators. Read them carefully and address them step by step."
                            ),
                        ],
                    },
                )

    return rendevous


# ===========================================
#                 Shared Prompts
# ===========================================
prompt_asking_agent_to_write_code_iteratively = """
To accomplish this, you will first think for a while and explain what you're going to do in your response. Then write Python code in your response using a code block (```python\n[code]\n```). You will write the code, run it, and then check the result. Repeat this process until the data is fully cleaned. The code should be as minimal as possible—avoid writing a long script that tries to handle everything at once. If there's an error early in a long script, the rest of the code won't run and may become irrelevant or require rewriting, which leads to wasted effort. A step-by-step approach—writing short, focused code, executing it, checking the output, and then proceeding—is more reliable. Think of each response as a Jupyter Notebook cell—each should contain semantically independent code.

Here is an example:
<example>
Hmm okay, so the column 'color' has missing values. But it's not that many. I'll simply fill them with the most frequent value.

```python
df.fillna({'color': df['color'].mode()[0]})
```
</example>
""".strip()

samples_of_using_done = """
<correct_example>
DONE
</correct_example>

<incorrect_example>
Okay this is the last code block. Let's drop the null values.

```python
df.dropna()
```

DONE
</incorrect_example>

<incorrect_example>
Okay! It looks all good now. We can move on to the next step.

DONE
</incorrect_example>
""".strip()


# ===========================================
#                 Retry Policy
# ===========================================
def retry_on(exc: Exception) -> bool:
    print(f"[Retry policy captured an exception]\n{exc}\n")
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    ):
        return False
    return True


# Retry Policy
retry_policy = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=128.0,
    max_attempts=3,
    jitter=True,
    retry_on=retry_on,
)
