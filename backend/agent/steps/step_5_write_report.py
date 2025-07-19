from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langgraph.graph import START, StateGraph
from langgraph.config import get_stream_writer

from agent.llms import reasoning_model_large
from agent.state import OverallState
from agent.common import retry_policy
from agent.utils import get_variable_descriptions


def agent(state: OverallState):
    writer = get_stream_writer()
    writer(
        {
            "stream_message": "📄 Preparing final comprehensive report...",
            "current_step": 5,
        }
    )

    input_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """
You are a data analyst agent. Currently you are in the final step of the data analysis process: Write Report. In this step your goal is to synthesize all the findings from previous analysis steps into a comprehensive, cohesive final report that directly answers the original objective.

# Original Objective: {objective}

# Reports from Previous Steps:
{step_reports}

---

Your task is to:
1. Review all the findings from the data exploration and analysis steps
2. Synthesize the key insights that directly address the original objective
3. Write a comprehensive final report that includes:
   - Executive summary
   - Key findings
   - Data insights and patterns discovered
   - How to came to the conclusion. If any model or scoring system is used, you should explain the details of the model such as what features are used, what weights are assigned to each feature, etc. You could include the actual formula or code here.
   - Conclusions that answer the original objective
   - Any limitations or caveats

The report should be well-structured, professional, and accessible to both technical and non-technical stakeholders. The report should be in markdown format.
    """.strip()
            ),
            HumanMessagePromptTemplate.from_template(
                """
Please write the final comprehensive report based on all the analysis conducted. Don't say "Sure I will write the report" or anything like that. Just return the report.

Here are final variables created from the analysis. You can refer to this when filling the details of the report. Especially if the user asked for specific data such as the top 10 investment opportunities, you should use data from these variables.

{final_variables}

Don't forget to use markdown format for the report!
    """.strip()
            ),
        ]
    ).format_messages(
        objective=state.objective,
        step_reports="\n\n".join(
            [
                f"<step_{step.order}>\n{step.report}\n</step_{step.order}>"
                for step in state.steps
                if step.report and len(step.report) > 10
            ]
        ),
        final_variables=get_variable_descriptions(state.variables, truncate=False),
    )

    writer({"oneline_message": "✍️ Generating final report..."})
    response = (reasoning_model_large | StrOutputParser()).invoke(input_messages)

    writer({"oneline_message": "🎉 Final report completed!"})

    return {
        "final_report": response,
    }


g = StateGraph(OverallState)

g.add_edge(START, agent.__name__)

g.add_node(agent, retry=retry_policy)

g = g.compile()

with open("step_5_write_report.png", "wb") as f:
    f.write(g.get_graph().draw_mermaid_png())
