from google.adk.agents.llm_agent import Agent
import requests
from typing import List, Dict, Optional

def get_github_stars(username: str, max_repos: int = 50) -> dict:
    """
    Fetches starred repositories for a GitHub user.
    
    Args:
        username: GitHub username
        max_repos: Maximum number of starred repos to fetch (default 50)
    
    Returns:
        Dictionary with status and list of starred repositories
    """
    try:
        url = f"https://api.github.com/users/{username}/starred"
        headers = {"Accept": "application/vnd.github.v3+json"}
        params = {"per_page": min(max_repos, 100)}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        stars = response.json()
        
        # Extract relevant info
        repos = []
        for repo in stars[:max_repos]:
            repos.append({
                "name": repo["full_name"],
                "description": repo["description"] or "No description",
                "language": repo["language"] or "Not specified",
                "stars": repo["stargazers_count"],
                "url": repo["html_url"],
                "topics": repo.get("topics", [])
            })
        
        return {
            "status": "success",
            "username": username,
            "total_fetched": len(repos),
            "repositories": repos
        }
    
    except requests.exceptions.RequestException as e:
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
    return {
        "status": "success",
        "description": project_description,
        "message": "Project context received. Use this to filter relevant starred repos."
    }


def search_stars_by_keyword(username: str, keyword: str, max_repos: int = 50) -> dict:
    """
    Searches through starred repos for a specific keyword in name, description, or topics.
    
    Args:
        username: GitHub username
        keyword: Keyword to search for
        max_repos: Maximum repos to search through
    
    Returns:
        Dictionary with matching repositories
    """
    try:
        stars_result = get_github_stars(username, max_repos)
        
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


# Create the agent
root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='root_agent',  # name='github_stars_agent',
    description="Analyzes your GitHub stars to find repositories relevant to your current project.",
    instruction="""You are a helpful assistant that analyzes GitHub starred repositories to help developers find relevant tools and libraries for their projects.

When a user provides their GitHub username and describes their project:

1. First, use get_github_stars to fetch their starred repositories
2. Use analyze_project_needs to understand what they're building
3. Analyze which starred repos are most relevant to their project based on:
   - Technology stack mentioned
   - Problem domain
   - Specific features or capabilities needed
4. Present the top 5-10 most relevant repositories with:
   - Repository name and link
   - Why it's relevant to their project
   - Specific use cases or features that match their needs
   - Language and popularity (stars)

You can also use search_stars_by_keyword to quickly find repos matching specific technologies.

Be conversational, helpful, and provide practical integration suggestions. Focus on repos that would actually save them development time.

Example interaction:
User: "My username is octocat and I'm building a real-time voice AI agent with Python"
You: *fetch stars, then analyze and respond with relevant repos like FastAPI, WebRTC libraries, speech-to-text tools, etc.*
""",
    tools=[get_github_stars, analyze_project_needs, search_stars_by_keyword],
)


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