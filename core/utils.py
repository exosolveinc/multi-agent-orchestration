
import re

def clean_json_response(content: str) -> str:
    """
    Clean JSON response from LLM by removing markdown code blocks.
    
    Args:
        content: The raw string response from LLM
        
    Returns:
        Cleaned string containing just the JSON content
    """
    # Remove markdown code blocks
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    
    # Also handle case where it might be wrapped in just ```
    return content.strip()
