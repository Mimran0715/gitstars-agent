Git Stars Agent helps you turn your starred GitHub repos into practical project recommendations.

It is built with Google Agent Development Kit and exposes tools that:

- fetch a user's starred repositories from GitHub
- extract useful search terms from a project description
- keyword-search stars by name, description, language, and topics
- rank starred repositories against a project so the agent can recommend what is actually useful

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For higher GitHub API rate limits, create a token and pass it to the tools as `github_token`.

## Run With ADK

From the parent directory:

```bash
adk web
```

Then ask the agent something like:

> My GitHub username is `octocat`. I am building a Python voice AI agent with realtime audio, FastAPI, and a React dashboard. Which of my starred repos are useful?

The agent should call `recommend_starred_repos`, then explain the best matches with links, reasons, and suggested use cases.

## Main Tool

`recommend_starred_repos` is the highest-level tool:

```python
recommend_starred_repos(
    username="octocat",
    project_description="A Python voice AI agent with realtime audio and a React dashboard",
    max_repos=100,
    top_k=10,
)
```

It returns ranked recommendations with:

- score
- matched project terms
- match reason
- suggested use
- repo metadata such as language, stars, topics, license, and last push time
