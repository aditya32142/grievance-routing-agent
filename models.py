"""
Data models for the Grievance Routing Environment.
"""
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional

class GrievanceRoutingAction(Action):
    """Action: agent decides department, priority, and action for a complaint."""
    department: str = Field(..., description="Government department: sanitation, electricity, water, roads, health, police")
    priority: str = Field(..., description="Priority level: low, medium, high, critical")
    action: str = Field(..., description="Action to take: log_complaint, send_team, escalate, close_resolved")
    reasoning: Optional[str] = Field(default=None, description="Optional reasoning (bonus reward in hard mode)")

class GrievanceRoutingObservation(Observation):
    """Observation: the citizen complaint the agent must route."""
    complaint_id: int = Field(default=0, description="Unique complaint ID")
    complaint_text: str = Field(default="", description="The citizen complaint text")
    difficulty: str = Field(default="easy", description="Task difficulty: easy, medium, hard")
    reward: float = Field(default=0.0, description="Reward from last action")
    done: bool = Field(default=False, description="Episode complete?")
