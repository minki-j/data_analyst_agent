import os
import json
import pickle
import pandas as pd
from typing import Union, Any, Tuple

from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Result

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.serde.base import SerializerProtocol

from agent.state import Step, Data, DataType


MAX_SAMPLE_ITEMS = 30
MAX_SAMPLE_LENGTH = 10000


class CustomSerializer(SerializerProtocol):
    """Custom serializer that handles pandas DataFrames and other objects."""

    def dumps(self, obj: Any) -> bytes:
        if isinstance(obj, pd.DataFrame):
            # For DataFrames, we'll pickle them with a special type marker
            return pickle.dumps(("DataFrame", obj.to_dict()))
        return pickle.dumps(obj)

    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        if isinstance(obj, pd.DataFrame):
            # For DataFrames, we'll pickle them with a special type marker
            return "DataFrame", pickle.dumps(obj.to_dict())
        return "pickle", pickle.dumps(obj)

    def loads(self, data: bytes) -> Any:
        obj = pickle.loads(data)
        if isinstance(obj, tuple) and obj[0] == "DataFrame":
            # If it's our special DataFrame format, reconstruct it
            return pd.DataFrame.from_dict(obj[1])
        return obj

    def loads_typed(self, data: Tuple[str, bytes]) -> Any:
        type_str, bytes_data = data
        if type_str == "DataFrame":
            # If it's a DataFrame type, reconstruct it
            return pd.DataFrame.from_dict(pickle.loads(bytes_data))
        return pickle.loads(bytes_data)


def get_current_step(plan: list[Step]) -> Union[Step, None]:
    not_completed_steps = [step for step in plan if not step.completed]
    if not not_completed_steps:
        return None
    return min(not_completed_steps, key=lambda step: step.order)


def truncate_string_middle(s, max_length):
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= max_length:
        return s
    half = max_length // 2
    return f"{s[:half]} ... [TRUNCATED, {len(s) - max_length} chars omitted] ... {s[-half:]}"


def get_e2b_sandbox() -> Sandbox:
    """
    Initializes and returns an E2B sandbox instance.

    Returns:
        Sandbox: The initialized E2B Sandbox instance

    Raises:
        Exception: If E2B_PYTHON_AGENT_TEMPLATE_ID environment variable is not set
    """
    e2b_sandbox_template_id = os.getenv("E2B_PYTHON_AGENT_TEMPLATE_ID")
    if e2b_sandbox_template_id is None:
        raise Exception("E2B_PYTHON_AGENT_TEMPLATE_ID is not set")

    return Sandbox(
        template=e2b_sandbox_template_id, timeout=60 * 10, request_timeout=60
    )


def upload_file_to_e2b_sandbox(
    sandbox: Sandbox,
    variable: Union[pd.DataFrame, list, dict, tuple, str],
    filename_key: str,
) -> str:
    """
    Uploads data to an existing E2B sandbox.

    Args:
        sandbox: The E2B Sandbox instance to upload to
        data: The data to write to the sandbox. Can be:
            - pandas DataFrame (saved as CSV)
            - list, dict, or tuple (saved as JSON)
        filename_key: The filename/key to use for the file

    Returns:
        str: The file path where the file was written

    Raises:
        ValueError: If data type is not supported
    """
    dataframe_dir = os.getenv("E2B_VARIABLES_DIR")
    if dataframe_dir is None:
        print("E2B_VARIABLES_DIR is not set. Using /tmp as fallback.")
        dataframe_dir = "/tmp"  # fallback directory

    # Handle different data types
    if isinstance(variable, pd.DataFrame):
        # For DataFrames, save as CSV
        file_content = variable.to_csv()
        if not filename_key.endswith(".csv"):
            filename_key = f"{filename_key}.csv"
    elif isinstance(variable, (list, dict, tuple)):
        # For list, dict, tuple, save as JSON
        try:
            file_content = json.dumps(variable, indent=2, default=str)
            if not filename_key.endswith(".json"):
                filename_key = f"{filename_key}.json"
        except (TypeError, ValueError) as e:
            # If JSON serialization fails, fall back to pickle
            file_content = pickle.dumps(variable).decode(
                "latin-1"
            )  # Use latin-1 for binary safety
            if not filename_key.endswith(".pkl"):
                filename_key = f"{filename_key}.pkl"
    elif isinstance(variable, str):
        file_content = variable
        if not filename_key.endswith(".txt"):
            filename_key = f"{filename_key}.txt"
    else:
        raise ValueError(
            f"Unsupported data type: {type(variable)}. Supported types: DataFrame, list, dict, tuple"
        )

    file_path = os.path.join(dataframe_dir, filename_key)
    sandbox.files.write(file_path, file_content)

    return file_path


def get_code_to_load_uploaded_file(variables: list[Data], file_paths: list[str]) -> str:
    code = ["import pandas as pd", "import json"]
    for variable, file_path in zip(variables, file_paths):
        if variable.type == DataType.DATAFRAME:
            code.append(f"{variable.key} = pd.read_csv('{file_path}')")
        elif variable.type == DataType.JSON:
            code.append(
                f"with open('{file_path}', 'r') as f: {variable.key} = json.load(f)"
            )
        elif variable.type == DataType.TEXT or variable.type == DataType.STRING:
            code.append(f"{variable.key} = open('{file_path}', 'r').read()")
        # elif variable.type == DataType.PNG:
        #     code.append(f"{variable.key} = plt.imread('{file_path}')")
        else:
            raise ValueError(
                f"Unsupported data type: {type(variable)}. Supported types: DataFrame, list, dict, tuple"
            )
    return "\n".join(code)


def _get_sample(variable: Data, truncate: bool = True) -> str:
    if variable.type == DataType.DATAFRAME:
        if truncate:
            return variable.value.head(3).to_string()
        else:
            return variable.value.head(MAX_SAMPLE_ITEMS).to_string()
    elif variable.type == DataType.JSON:
        dumped_json = json.dumps(variable.value, indent=2, default=str)
        if truncate:
            return truncate_string_middle(dumped_json, 1000)
        else:
            return truncate_string_middle(dumped_json, MAX_SAMPLE_LENGTH)
    elif variable.type == DataType.TEXT:
        if truncate:
            return truncate_string_middle(variable.value, 1000)
        else:
            return truncate_string_middle(variable.value, MAX_SAMPLE_LENGTH)
    else:
        raise ValueError(
            f"Unsupported data type: {type(variable)}. Supported types: DataFrame, list, dict, tuple"
        )


def _convert_periods(obj):
    if isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_periods(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_periods(i) for i in obj]
    return obj


def parse_e2b_execution_results(results: list[Result]) -> list[tuple[Any, DataType]]:
    parsed_results = []
    for result in results:
        formats = result.formats()
        if "data" in formats:
            try:
                value = pd.DataFrame(result.data)
                value = value.map(lambda x: _convert_periods(x))
                parsed_results.append((value, DataType.DATAFRAME))
            except Exception as e:
                print(f"Error converting data to pandas DataFrame: {e}")
                raise ValueError(f"Failed to convert data to DataFrame: {e}")
        elif "json" in formats:
            try:
                if result.json is not None:
                    parsed_results.append((result.json, DataType.JSON))
                else:
                    raise Exception("No json data")
            except Exception as e:
                print(f"Error processing json data: {e}")
                raise ValueError(f"Failed to process json data: {e}")
        elif "text" in formats:
            parsed_results.append(
                (truncate_string_middle(result.text, 1000), DataType.TEXT)
            )
        elif "png" in formats:
            parsed_results.append((result.png, DataType.PNG))
        else:
            raise ValueError(f"Unsupported data type: {formats}")

    return parsed_results


def format_results(parsed_results: list[tuple[Any, DataType]]) -> str:
    formatted_values = []
    for value, data_type in parsed_results:
        if data_type == DataType.DATAFRAME:
            # For DataFrames, show first few rows and dimensions
            formatted_values.append(
                f"DataFrame ({value.shape[0]} rows x {value.shape[1]} columns):\n{value.head(3).to_string()}"
            )
        elif data_type == DataType.JSON:
            # For JSON, truncate if too large
            formatted_values.append(truncate_string_middle(str(value), 1000))
        else:
            # For other types, truncate if too large
            formatted_values.append(truncate_string_middle(str(value), 1000))
    return "\n".join(formatted_values)


def get_variable_descriptions(variables: list[Data], truncate: bool = True) -> str:
    return "\n".join(
        [
            f"<variable_name>{variable.key}</variable_name>\n<description>{variable.description}</description>\n<sample_data>{_get_sample(variable, truncate)}</sample_data>"
            for variable in variables
        ]
    )


def get_dataframe_info(variables: list[Data]) -> str:
    for variable in variables:
        if variable.type == DataType.DATAFRAME:
            df = variable.value
            info_sections = []

            # Basic Information
            info_sections.append(f"Shape: {df.shape} (rows × columns)")

            # Data Types and Non-null Counts
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                null_count = len(df) - non_null
                null_percentage = (null_count / len(df)) * 100 if len(df) > 0 else 0
                info_sections.append(
                    f"{col}: {dtype} | Non-null: {non_null} | Missing: {null_count} ({null_percentage:.1f}%)"
                )

            # Descriptive Statistics for Numeric Columns
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                desc_stats = df[numeric_cols].describe()
                info_sections.append(desc_stats.to_string())

            # Categorical Columns Summary
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    most_frequent = df[col].mode()
                    if len(most_frequent) > 0:
                        most_frequent_value = most_frequent.iloc[0]
                        most_frequent_count = df[col].value_counts().iloc[0]
                        info_sections.append(
                            f"{col}: {unique_count} unique values | Most frequent: '{most_frequent_value}' ({most_frequent_count} times)"
                        )
                    else:
                        info_sections.append(f"{col}: {unique_count} unique values")

            return f"<dataframe_info>{'\n\n'.join(info_sections)}</dataframe_info>"
    return ""


string_output_parser = StrOutputParser()


def extract_code_block(response: BaseMessage) -> str | None:
    content = string_output_parser.invoke(response)

    if "DONE" in content:
        return None

    try:
        code_block = content.split("```python")[1].split("```")[0].strip()
        return code_block
    except Exception as e:
        # print(f"Error parsing code block: {e}\n\nCode Block:\n{content}")
        return None
