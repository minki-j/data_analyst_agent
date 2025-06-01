from typing import Any
from enum import Enum
from pydantic import BaseModel, Field


# ===========================================
#                Data Models
# ===========================================
class Step(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "use_enum_values": True}

    order: int
    name: str
    description: str
    completed: bool = Field(default=False)
    report: str = Field(default="")


class DataType(str, Enum):
    DATAFRAME = "dataframe"
    JSON = "json"
    TEXT = "text"
    STRING = "string"
    PNG = "png"


class Data(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "use_enum_values": True}

    key: str
    type: DataType
    description: str
    value: Any


# ===========================================
#              REDUCER FUNCTIONS
# ===========================================
def extend_list(original: list, new: Any):
    if not isinstance(new, list):
        new = [new]
    if len(new) == 1 and new[0] == "RESET_LIST":
        return []
    original.extend(new)
    return original


# ===========================================
#               DEFAULT VALUES
# ===========================================
def get_default_steps():
    """Function to create default steps"""
    return [
        Step(
            order=1,
            name="Define the Objective",
            description="Understand the problem and set goals",
            completed=False,
            report="",
        ),
        Step(
            order=2,
            name="Data Cleaning",
            description="Handle missing data, fix errors, and filter irrelevant data",
            completed=False,
            report="",
        ),
        Step(
            order=3,
            name="Data Exploration",
            description="Summarize data and find anomalies/outliers",
            completed=False,
            report="",
        ),
        Step(
            order=4,
            name="Data Analysis & Visualization",
            description="",
            completed=False,
            report="",
        ),
        Step(
            order=5,
            name="Write Report",
            description="",
            completed=False,
            report="",
        ),
    ]


# ===========================================
#                    STATE
# ===========================================
class InputState(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "use_enum_values": True}

    objective: str
    variables: list[Data]
    skip_define_objective_step: bool = Field(default=False)
    use_human_in_the_loop: bool = Field(default=False)


class OutputState(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "use_enum_values": True}

    final_report: str = Field(default="")


class OverallState(InputState, OutputState):
    model_config = {"arbitrary_types_allowed": True, "use_enum_values": True}

    steps: list[Step] = Field(default_factory=get_default_steps)
    sandbox_id: str = Field(default="")
