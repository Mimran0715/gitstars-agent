from collections import Counter
from typing import Any, Dict, List, Optional, Set

try:
    import requests
except ModuleNotFoundError:
    requests = None

try:
    from google.adk.agents.llm_agent import Agent
except ModuleNotFoundError:
    Agent = None


GITHUB_API_URL = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS = 20

STOP_WORDS = {
    "about",
    "agent",
    "also",
    "and",
    "app",
    "are",
    "build",
    "building",
    "can",
    "current",
    "for",
    "from",
    "have",
    "into",
    "like",
    "make",
    "need",
    "needs",
    "project",
    "repo",
    "repos",
    "that",
    "the",
    "this",
    "tool",
    "use",
    "using",
    "want",
    "with",
    "would",
    "your",
}


DOMAIN_HINTS = {
    "ai": {"llm", "agent", "agents", "openai", "gemini", "rag", "vector", "embedding", "prompt"},
    "backend": {"api", "fastapi", "django", "flask", "server", "auth", "database", "postgres"},
    "cli": {"cli", "terminal", "command", "shell", "tui"},
    "data": {"data", "etl", "pipeline", "analytics", "warehouse", "sql", "duckdb"},
    "frontend": {"react", "nextjs", "vue", "svelte", "ui", "css", "tailwind", "component"},
    "github": {"github", "git", "repo", "repository", "stars", "issues", "pull", "actions"},
    "mobile": {"ios", "android", "react-native", "swift", "kotlin", "flutter"},
    "testing": {"test", "testing", "pytest", "playwright", "jest", "ci"},
}


def _extract_terms(text: str) -> Set[str]:
    """Extract searchable project terms without adding a heavyweight NLP dependency."""
    normalized = "".join(ch.lower() if ch.isalnum() or ch in {"-", "_", "."} else " " for ch in text)
    raw_terms = [term.strip("-_.") for term in normalized.split()]
    return {term for term in raw_terms if len(term) > 2 and term not in STOP_WORDS}


def _repo_search_blob(repo: Dict[str, Any]) -> str:
    fields = [
        repo.get("name", ""),
        repo.get("description", ""),
        repo.get("language", ""),
        " ".join(repo.get("topics", [])),
    ]
    return " ".join(value for value in fields if value).lower()


def _score_repo(repo: Dict[str, Any], project_terms: Set[str]) -> Dict[str, Any]:
    blob = _repo_search_blob(repo)
    repo_topics = {topic.lower() for topic in repo.get("topics", [])}
    repo_name = repo.get("name", "").lower()
    repo_language = (repo.get("language") or "").lower()

    matched_terms: List[str] = []
    score = 0.0

    for term in sorted(project_terms):
        if term in repo_name:
            score += 6
            matched_terms.append(term)
        elif term in repo_topics:
            score += 5
            matched_terms.append(term)
        elif term == repo_language:
            score += 4
            matched_terms.append(term)
        elif term in blob:
            score += 2
            matched_terms.append(term)

    stars = repo.get("stars", 0) or 0
    if stars >= 50_000:
        score += 5
    elif stars >= 10_000:
        score += 4
    elif stars >= 1_000:
        score += 3
    elif stars >= 100:
        score += 1

    if repo.get("archived"):
        score -= 4

    if repo.get("pushed_at"):
        score += 1

    unique_matches = sorted(set(matched_terms))
    return {
        "score": round(score, 2),
        "matched_terms": unique_matches[:12],
        "match_reason": _build_match_reason(repo, unique_matches),
    }


def _build_match_reason(repo: Dict[str, Any], matched_terms: List[str]) -> str:
    if not matched_terms:
        return "General-purpose starred repo; review manually for fit."

    language = repo.get("language") or "the repo's stack"
    topics = repo.get("topics", [])
    topic_text = f" and topics like {', '.join(topics[:3])}" if topics else ""
    term_text = ", ".join(matched_terms[:5])
    return f"Matches project terms ({term_text}) through {language}{topic_text}."


def _format_repo(repo: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": repo["full_name"],
        "description": repo["description"] or "No description",
        "language": repo["language"] or "Not specified",
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "open_issues": repo["open_issues_count"],
        "archived": repo["archived"],
        "url": repo["html_url"],
        "topics": repo.get("topics", []),
        "license": (repo.get("license") or {}).get("spdx_id"),
        "pushed_at": repo.get("pushed_at"),
    }


def _github_headers(token: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_starred_repos(username: str, max_repos: int, token: Optional[str] = None) -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("The 'requests' package is required to fetch GitHub stars. Run: pip install -r requirements.txt")

    repos: List[Dict[str, Any]] = []
    page = 1

    while len(repos) < max_repos:
        response = requests.get(
            f"{GITHUB_API_URL}/users/{username}/starred",
            headers=_github_headers(token),
            params={"per_page": min(100, max_repos - len(repos)), "page": page},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        page_repos = response.json()
        if not page_repos:
            break

        repos.extend(_format_repo(repo) for repo in page_repos)
        page += 1

    return repos[:max_repos]


def get_github_stars(username: str, max_repos: int = 50, github_token: Optional[str] = None) -> dict:
    """
    Fetches starred repositories for a GitHub user.
    
    Args:
        username: GitHub username
        max_repos: Maximum number of starred repos to fetch (default 50)
        github_token: Optional GitHub token for higher rate limits or private authenticated access
    
    Returns:
        Dictionary with status and list of starred repositories
    """
    try:
        repos = _fetch_starred_repos(username, max_repos, github_token)
        return {
            "status": "success",
            "username": username,
            "total_fetched": len(repos),
            "repositories": repos
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch stars: {str(e)}"
        }


def analyze_project_needs(project_description: str) -> dict:
    """
    Analyzes a project description to extract key technologies and needs.
    This is a helper function - the LLM will use this to understand the project.
    
    Args:
        project_description: Description of the current project
    
    Returns:
        Dictionary with extracted project context
    """
    terms = sorted(_extract_terms(project_description))
    primary_signals = Counter()
    for domain, hints in DOMAIN_HINTS.items():
        primary_signals[domain] = len(set(terms).intersection(hints))

    likely_domains = [domain for domain, _ in primary_signals.most_common(4) if primary_signals[domain] > 0]

    return {
        "status": "success",
        "description": project_description,
        "search_terms": terms[:30],
        "likely_domains": likely_domains,
        "message": "Project context received. Use these terms and domains to filter relevant starred repos."
    }


def search_stars_by_keyword(
    username: str,
    keyword: str,
    max_repos: int = 50,
    github_token: Optional[str] = None,
) -> dict:
    """
    Searches through starred repos for a specific keyword in name, description, or topics.
    
    Args:
        username: GitHub username
        keyword: Keyword to search for
        max_repos: Maximum repos to search through
        github_token: Optional GitHub token for higher rate limits or private authenticated access
    
    Returns:
        Dictionary with matching repositories
    """
    try:
        stars_result = get_github_stars(username, max_repos, github_token)
        
        if stars_result["status"] == "error":
            return stars_result
        
        keyword_lower = keyword.lower()
        matching_repos = []
        
        for repo in stars_result["repositories"]:
            # Search in name, description, language, and topics
            if (keyword_lower in repo["name"].lower() or
                keyword_lower in repo["description"].lower() or
                keyword_lower in (repo["language"] or "").lower() or
                any(keyword_lower in topic.lower() for topic in repo["topics"])):
                matching_repos.append(repo)
        
        return {
            "status": "success",
            "keyword": keyword,
            "total_matches": len(matching_repos),
            "repositories": matching_repos
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }


def recommend_starred_repos(
    username: str,
    project_description: str,
    max_repos: int = 100,
    top_k: int = 10,
    include_archived: bool = False,
    github_token: Optional[str] = None,
) -> dict:
    """
    Scores a user's starred repos against a project description and returns the best matches.

    Args:
        username: GitHub username
        project_description: What the user is building and what they need help with
        max_repos: Maximum starred repos to inspect
        top_k: Number of recommendations to return
        include_archived: Whether archived repositories can be recommended
        github_token: Optional GitHub token for higher rate limits or private authenticated access

    Returns:
        Dictionary with ranked recommendations and project analysis
    """
    try:
        project_analysis = analyze_project_needs(project_description)
        project_terms = set(project_analysis["search_terms"])
        repos = _fetch_starred_repos(username, max_repos, github_token)

        scored_repos = []
        for repo in repos:
            if repo["archived"] and not include_archived:
                continue

            match = _score_repo(repo, project_terms)
            if match["score"] <= 0:
                continue

            scored_repos.append({
                **repo,
                **match,
                "suggested_use": _suggest_use(repo, match["matched_terms"]),
            })

        scored_repos.sort(key=lambda repo: (repo["score"], repo["stars"]), reverse=True)

        return {
            "status": "success",
            "username": username,
            "project_analysis": project_analysis,
            "total_fetched": len(repos),
            "total_ranked": len(scored_repos),
            "recommendations": scored_repos[:top_k],
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to recommend repos: {str(e)}"
        }


def _suggest_use(repo: Dict[str, Any], matched_terms: List[str]) -> str:
    if not matched_terms:
        return "Open the README and evaluate whether it solves a nearby problem."

    repo_name = repo.get("name", "this repo")
    terms = ", ".join(matched_terms[:3])
    return f"Review {repo_name} as a candidate for the parts of your project involving {terms}."


if Agent is not None:
    root_agent = Agent(
        model='gemini-2.0-flash-exp',
        name='root_agent',  # name='github_stars_agent',
        description="Analyzes your GitHub stars to find repositories relevant to your current project.",
        instruction="""You are a helpful assistant that analyzes GitHub starred repositories to help developers find relevant tools and libraries for their projects.

When a user provides their GitHub username and describes their project:

1. Prefer recommend_starred_repos to fetch, analyze, and rank their starred repositories in one pass
2. Use analyze_project_needs when you need to clarify or summarize what they're building
3. Use get_github_stars or search_stars_by_keyword for follow-up exploration
4. Analyze which starred repos are most relevant to their project based on:
   - Technology stack mentioned
   - Problem domain
   - Specific features or capabilities needed
5. Present the top 5-10 most relevant repositories with:
   - Repository name and link
   - Why it's relevant to their project
   - Specific use cases or features that match their needs
   - Language and popularity (stars)
   - Any caveats such as archived repos, inactive projects, or license concerns

You can also use search_stars_by_keyword to quickly find repos matching specific technologies.

Be conversational, helpful, and provide practical integration suggestions. Focus on repos that would actually save them development time.
If the ranked results look weak, say that and suggest more specific project details or keywords.

Example interaction:
User: "My username is octocat and I'm building a real-time voice AI agent with Python"
You: *fetch stars, then analyze and respond with relevant repos like FastAPI, WebRTC libraries, speech-to-text tools, etc.*
""",
        tools=[get_github_stars, analyze_project_needs, search_stars_by_keyword, recommend_starred_repos],
    )
else:
    root_agent = None


# from google.adk.agents.llm_agent import Agent

# # Mock tool implementation
# def get_current_time(city: str) -> dict:
#     """Returns the current time in a specified city."""
#     return {"status": "success", "city": city, "time": "10:30 AM"}

# root_agent = Agent(
#     model='gemini-3-flash-preview',
#     name='root_agent',
#     description="Tells the current time in a specified city.",
#     instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
#     tools=[get_current_time],
# )
