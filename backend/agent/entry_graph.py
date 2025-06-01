from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import OverallState, InputState, OutputState
from .utils import get_current_step, CustomSerializer

from .steps.step_1_define_objective import g as step_1_define_objective
from .steps.step_2_data_cleaning import g as step_2_data_cleaning
from .steps.step_3_data_exploration import g as step_3_data_exploration
from .steps.step_4_data_analysis import g as step_4_data_analysis
from .steps.step_5_write_report import g as step_5_write_report


def check_if_skip_any_step(state: OverallState):
    if state.skip_define_objective_step:
        for step in state.steps:
            if step.order == 1:
                step.completed = True
                break
        return {"steps": state.steps}
    else:
        return {}


def stage_router(state: OverallState):
    current_step = get_current_step(state.steps)

    if current_step is None:
        return END

    if current_step.order == 1:
        return "step_1_define_objective"
    elif current_step.order == 2:
        return "step_2_data_cleaning"
    elif current_step.order == 3:
        return "step_3_data_exploration"
    elif current_step.order == 4:
        return "step_4_data_analysis"
    elif current_step.order == 5:
        return "step_5_write_report"
    else:
        print(f"Unknown step: {current_step}")
        return END


g = StateGraph(OverallState, input=InputState, output=OutputState)
g.add_edge(START, "check_if_skip_any_step")

g.add_node(check_if_skip_any_step)
g.add_conditional_edges(
    "check_if_skip_any_step",
    stage_router,
    [
        "step_1_define_objective",
        "step_2_data_cleaning",
        "step_3_data_exploration",
        "step_4_data_analysis",
        "step_5_write_report",
        END,
    ],
)

g.add_node("step_1_define_objective", step_1_define_objective)
g.add_edge("step_1_define_objective", "step_2_data_cleaning")

g.add_node("step_2_data_cleaning", step_2_data_cleaning)
g.add_edge("step_2_data_cleaning", "step_3_data_exploration")

g.add_node("step_3_data_exploration", step_3_data_exploration)
g.add_edge("step_3_data_exploration", "step_4_data_analysis")

g.add_node("step_4_data_analysis", step_4_data_analysis)
g.add_edge("step_4_data_analysis", "step_5_write_report")

g.add_node("step_5_write_report", step_5_write_report)
g.add_edge("step_5_write_report", END)

serializer = CustomSerializer()

g = g.compile(
    checkpointer=MemorySaver(serde=serializer),
    # interrupt_after=[
    #     "step_1_define_objective",
    #     "step_2_data_cleaning",
    #     "step_3_data_exploration",
    #     "step_4_data_analysis",
    #     "step_5_write_report",
    # ],
)
