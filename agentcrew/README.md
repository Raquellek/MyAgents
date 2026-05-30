# Agentcrew Paper Crew

`agentcrew` is now configured as a paper-writing CrewAI project with five
collaborating agents:

- `manager`: scopes the paper and defines the execution plan
- `researcher`: gathers evidence and supporting sources
- `coder`: prepares technical details, pseudocode, and evaluation ideas
- `writer`: drafts the manuscript
- `critic`: reviews and rewrites the draft into the final paper

## Run

From the project root:

```bash
cd agentcrew
crewai run "Multi-agent systems for scientific writing"
```

If no topic is passed, the default topic is `AI LLMs`.

## Output

The final reviewed paper is written to:

```text
output/paper.md
```

## Environment

Set the model settings in `.env`. The current project is configured for a local
Ollama endpoint:

```env
MODEL=ollama/llama3.1
API_BASE=http://localhost:11434
```
