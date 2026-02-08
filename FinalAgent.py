"""
Travel Planning System - Final Project
Multi-Agent Intelligent Travel Planner using Google ADK

Implements:
- Session State Management
- Tool-based Agents
- Parallel Research
- Sequential Itinerary Pipeline
- Optimization Loop
- Security Layer with before_model_callback
"""

from google.adk.agents.llm_agent import Agent
from google.adk.tools import tool

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

SESSION_STATE = {
    "preferences": {},
    "history": []
}

# ---------------------------------------------------------------------------
# Security Layer - before_model_callback
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = ["dangerous border", "war zone", "illegal"]

def security_callback(prompt: str) -> str:
    """Blocks unsafe or forbidden user requests before model execution."""
    for word in FORBIDDEN_KEYWORDS:
        if word in prompt.lower():
            raise ValueError("Request blocked due to security policy.")
    return prompt

# ---------------------------------------------------------------------------
# Tools (Library-based Style)
# ---------------------------------------------------------------------------

@tool
def weather_tool(city: str) -> str:
    """Returns current weather conditions for a city."""
    return f"The weather in {city} is sunny and suitable for sightseeing."

@tool
def transport_tool(city: str) -> str:
    """Provides transportation guidance for the city."""
    return f"In {city}, metro and walking are the best transportation options."

@tool
def budget_tool(days: int, budget: int) -> str:
    """Estimates daily budget allocation."""
    return f"Estimated daily budget: {budget // days} Toman."

# ---------------------------------------------------------------------------
# Gatekeeper Agent
# ---------------------------------------------------------------------------

gatekeeper_agent = Agent(
    name="Gatekeeper_Agent",
    model="gemini-2.5-flash",
    description="Extracts user preferences and manages session state",
    instruction="""
Extract user preferences including destination, budget, interests,
constraints, and trip duration. Store them in the session state.
Perform initial security validation.
""",
    before_model_callback=security_callback
)

# ---------------------------------------------------------------------------
# Parallel Research Agents
# ---------------------------------------------------------------------------

attraction_agent = Agent(
    name="Attraction_Researcher",
    model="gemini-2.5-flash",
    instruction="Research and list major tourist attractions of the destination."
)

accommodation_agent = Agent(
    name="Accommodation_Researcher",
    model="gemini-2.5-flash",
    instruction="Find suitable hotels or accommodations within the user's budget."
)

service_agent = Agent(
    name="Service_Researcher",
    model="gemini-2.5-flash",
    instruction="Recommend local restaurants and food options."
)

# ---------------------------------------------------------------------------
# Sequential Itinerary Pipeline
# ---------------------------------------------------------------------------

structure_agent = Agent(
    name="Itinerary_Structurer",
    model="gemini-2.5-flash",
    instruction="Create a day-by-day itinerary framework."
)

detail_agent = Agent(
    name="Itinerary_Detailer",
    model="gemini-2.5-flash",
    instruction="Fill the itinerary framework with researched details."
)

logic_agent = Agent(
    name="Logic_Validator",
    model="gemini-2.5-flash",
    instruction="Check time consistency, feasibility, and logical flow."
)

# ---------------------------------------------------------------------------
# Optimization Loop
# ---------------------------------------------------------------------------

critic_agent = Agent(
    name="Plan_Critic",
    model="gemini-2.5-flash",
    instruction="Critically evaluate the itinerary from multiple aspects."
)

improver_agent = Agent(
    name="Plan_Improver",
    model="gemini-2.5-flash",
    instruction="Improve the plan based on critic feedback."
)

optimization_loop = [critic_agent, improver_agent]

# ---------------------------------------------------------------------------
# Root Agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="Main_Travel_Coordinator",
    model="gemini-2.5-flash",
    description="Coordinates all agents to generate a complete travel plan",
    instruction="""
Generate the final travel plan including:
1. Daily itinerary
2. Tourist attractions
3. Restaurant recommendations
4. Cost estimation
5. Transportation tips
6. Weather conditions
""",
    tools=[weather_tool, transport_tool, budget_tool],
    sub_agents=[
        gatekeeper_agent,
        attraction_agent,
        accommodation_agent,
        service_agent,
        structure_agent,
        detail_agent,
        logic_agent,
        critic_agent,
        improver_agent
    ]
)
