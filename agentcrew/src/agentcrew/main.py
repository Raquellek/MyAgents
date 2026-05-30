#!/usr/bin/env python
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep CrewAI's SQLite/appdirs storage inside the project so it remains writable
# in sandboxed, Snap, or otherwise restricted environments.
os.environ["XDG_DATA_HOME"] = str(PROJECT_ROOT / ".local" / "share")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentcrew.crew import Agentcrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def build_inputs() -> dict[str, str]:
    topic = " ".join(sys.argv[1:]).strip() or "AI LLMs"
    return {
        "topic": topic,
        "current_year": str(datetime.now().year),
    }


def run():
    """Run the crew."""
    try:
        Agentcrew().crew().kickoff(inputs=build_inputs())
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    try:
        Agentcrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs={
                "topic": " ".join(sys.argv[3:]).strip() or "AI LLMs",
                "current_year": str(datetime.now().year),
            },
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        Agentcrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and return the results."""
    try:
        Agentcrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs={
                "topic": " ".join(sys.argv[3:]).strip() or "AI LLMs",
                "current_year": str(datetime.now().year),
            },
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """Run the crew with trigger payload."""
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    payload_topic = ""
    if isinstance(trigger_payload, dict):
        payload_topic = str(trigger_payload.get("topic", "")).strip()

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": payload_topic or "AI LLMs",
        "current_year": str(datetime.now().year),
    }

    try:
        return Agentcrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


if __name__ == "__main__":
    run()
