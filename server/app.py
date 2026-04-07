"""
FastAPI application for the Grievance Routing Environment.
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies properly."
    ) from e

try:
    from models import GrievanceRoutingAction, GrievanceRoutingObservation
    from server.grievance_routing_environment import (
        GrievanceRoutingEnvironment,
        TASK_CONFIGS,
        grade_task,
        list_available_tasks,
    )
except ImportError:
    from ..models import GrievanceRoutingAction, GrievanceRoutingObservation
    from .grievance_routing_environment import (
        GrievanceRoutingEnvironment,
        TASK_CONFIGS,
        grade_task,
        list_available_tasks,
    )

from fastapi import HTTPException
from fastapi.responses import RedirectResponse
import uvicorn

# Create FastAPI app (FINAL WORKING)
app = create_app(
    GrievanceRoutingEnvironment,
    GrievanceRoutingAction,
    GrievanceRoutingObservation,
    env_name="grievance_routing",
    max_concurrent_envs=1
)

# Redirect root to API docs (since /web not supported)
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/tasks", tags=["grader"])
async def tasks():
    return {
        "tasks": list_available_tasks(),
        "count": len(TASK_CONFIGS),
    }


@app.get("/grade/{task_id}", tags=["grader"])
async def grade(task_id: str):
    if task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    return grade_task(task_id)


@app.get("/validate", tags=["grader"])
async def validate():
    task_results = [grade_task(task_id) for task_id in TASK_CONFIGS]
    all_scores_valid = all(0.0 < result["score"] < 1.0 for result in task_results)
    all_have_graders = all(result["grader"] for result in task_results)
    return {
        "valid": len(task_results) >= 3 and all_scores_valid and all_have_graders,
        "checks": {
            "min_tasks": len(task_results) >= 3,
            "all_have_graders": all_have_graders,
            "scores_in_open_interval": all_scores_valid,
        },
        "task_results": task_results,
        "env_name": "grievance_routing",
    }


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
