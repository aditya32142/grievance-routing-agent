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
    from server.grievance_routing_environment import GrievanceRoutingEnvironment
except ImportError:
    from ..models import GrievanceRoutingAction, GrievanceRoutingObservation
    from .grievance_routing_environment import GrievanceRoutingEnvironment

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


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
