import os
import json
import pandas as pd
from datetime import datetime

from pydantic import BaseModel
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect

from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from agent.state import Data, DataType
from agent.entry_graph import g

from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="Data Analyst Agent API", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    question: str
    city: str
    purpose: str
    rooms: int
    type: str
    budget: str
    topN: int
    investmentTimeline: int
    method: str
    additionalInfo: str
    skipDefineObjectiveStep: Optional[bool] = False
    useHumanInTheLoop: Optional[bool] = False
    data_file_path: Optional[str] = "./backend/run/melb_2bed.csv"


# Global storage for active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


def create_objective(request: AnalysisRequest) -> str:
    """Create objective string from user input"""
    objective = f"""
The user asked the following question:
{request.question}

And here are the details:
- purpose: {request.purpose}
- city: {request.city}
- rooms: {request.rooms}
- type: {request.type}
- budget: {request.budget}
- top_n: {request.topN}
"""

    if request.method:
        objective += f"- method: {request.method}\n"
    if request.additionalInfo:
        objective += f"- etc: {request.additionalInfo}\n"

    return objective.strip()


def create_graph_input(request: AnalysisRequest) -> Dict[str, Any]:
    """Create graph input from analysis request"""
    objective = create_objective(request)

    # Load the data
    data_path = request.data_file_path or "./run/melb_2bed.csv"
    if not os.path.isabs(data_path):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), data_path.lstrip("./")
        )

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Data file not found: {data_path}")

    return {
        "skip_define_objective_step": request.skipDefineObjectiveStep,
        "use_human_in_the_loop": request.useHumanInTheLoop,
        "variables": [
            Data(
                key="melbourne_housing_df",
                type=DataType.DATAFRAME,
                description="This dataframe contains the melbourne real estate sales data from 2016 to 2017.",
                value=df,
            )
        ],
        "objective": objective,
    }


@app.websocket("/agent/")
async def websocket_analysis(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis"""
    try:
        await websocket.accept()

        # Receive initial data from frontend
        data = await websocket.receive_json()

        form_data = data.get("form_data", "")

        if form_data:
            form_data = json.loads(form_data)
        else:
            await websocket.send_json({"error": "No input provided"})
            return

        # Create session ID
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create user input from the text
        request = AnalysisRequest(**form_data)

        try:
            graph_input = create_graph_input(request)
        except Exception as e:
            await websocket.send_json({"error": str(e)})
            return

        # Configure the graph
        config = RunnableConfig(
            {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 1000,
            }
        )

        # Store session info
        active_sessions[session_id] = {
            "status": "running",
            "config": config,
            "websocket": websocket,
        }

        # Stream the graph execution
        try:
            async for stream_mode, chunk in g.astream(
                graph_input,
                config=config,
                stream_mode=["custom", "updates"],
            ):
                print("--------------------------------")
                print("stream_mode", stream_mode)
                print("chunk", chunk)
                print("chunk type", type(chunk))
                if stream_mode == "custom":
                    if "oneline_message" in chunk:
                        await websocket.send_json(
                            {"oneline_message": chunk["oneline_message"]}
                        )
                        print("oneline_message sent via websocket")
                    if "current_step" in chunk:
                        await websocket.send_json(
                            {"current_step": chunk["current_step"]}
                        )
                        print("current_step sent via websocket")

            # Check if we need human input
            while True:
                try:
                    state = g.get_state_history(config)
                    latest_snapshot = next(state)
                    if latest_snapshot.interrupts:
                        message_to_user = latest_snapshot.interrupts[0].value[
                            "message_to_user"
                        ]
                        active_sessions[session_id]["status"] = "waiting_for_input"
                        active_sessions[session_id]["message_to_user"] = message_to_user

                        await websocket.send_json(
                            {
                                "summary": "Please answer this question!",
                                "content": message_to_user,
                                "requires_input": True,
                            }
                        )

                        # Wait for user response
                        print("waiting for user response")
                        user_response_data = await websocket.receive_json()
                        user_response = user_response_data.get("input", "")
                        print("user_response: ", user_response)

                        if user_response.lower() in ["q", "quit"]:
                            active_sessions[session_id]["status"] = "cancelled"
                            await websocket.send_json({"summary": "Analysis Cancelled"})
                            break

                        # Continue the graph execution
                        active_sessions[session_id]["status"] = "running"

                        async for stream_mode, chunk in g.astream(
                            Command(resume=user_response.strip()),
                            config=config,
                            stream_mode=["custom", "updates"],
                        ):
                            print("--------------------------------")
                            print("stream_mode", stream_mode)
                            print("chunk", chunk)
                            print("chunk type", type(chunk))
                            if stream_mode == "custom":
                                if "oneline_message" in chunk:
                                    await websocket.send_json(
                                        {"oneline_message": chunk["oneline_message"]}
                                    )
                                    print("oneline_message sent via websocket")
                                if "current_step" in chunk:
                                    await websocket.send_json(
                                        {"current_step": chunk["current_step"]}
                                    )
                                    print("current_step sent via websocket")

                    else:
                        print(
                            "Last snapshot doesn't have interrupts:\n", latest_snapshot
                        )
                        # Analysis completed
                        active_sessions[session_id]["status"] = "completed"
                        final_report = latest_snapshot.values.get("final_report", "")
                        active_sessions[session_id]["final_report"] = final_report

                        await websocket.send_json(
                            {
                                "summary": "Analysis Complete",
                                "content": final_report,
                                "completed": True,
                            }
                        )
                        break
                except Exception as e:
                    # Analysis completed or error
                    try:
                        state = g.get_state_history(config)
                        latest_snapshot = next(state)
                        active_sessions[session_id]["status"] = "completed"
                        final_report = latest_snapshot.values.get("final_report", "")
                        active_sessions[session_id]["final_report"] = final_report

                        await websocket.send_json({"final_report": final_report})
                    except Exception as inner_e:
                        await websocket.send_json(
                            {
                                "summary": "Error",
                                "content": f"An error occurred: {str(inner_e)}",
                            }
                        )
                    break

        except Exception as e:
            active_sessions[session_id]["status"] = "error"
            active_sessions[session_id]["error"] = str(e)
            await websocket.send_json(
                {
                    "summary": "Error",
                    "content": f"An error occurred during analysis: {str(e)}",
                }
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Data Analyst Agent API", "version": "0.1.0"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
