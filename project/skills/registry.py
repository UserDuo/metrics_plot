import inspect
import json
import os
from typing import Callable, Dict, Any, List, Optional
from project.skills.loader import SkillLoader

class Skill:
    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.parameters = self._generate_schema()

    def _generate_schema(self) -> Dict[str, Any]:
        """
        Generates a JSON schema for the function parameters.
        Simplified version for demonstration.
        """
        sig = inspect.signature(self.func)
        params = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = "string" # Default
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            
            params[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        return {
            "type": "object",
            "properties": params,
            "required": required
        }

    def to_tool_schema(self) -> Dict[str, Any]:
        """
        Returns the schema in a format compatible with LLM tool use (e.g., Claude/OpenAI).
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        # Initialize Contextual Skill Loader
        library_path = os.path.join(os.path.dirname(__file__), "library")
        self.loader = SkillLoader(library_path)
        
        # Register management tools
        self.register(self.list_contextual_skills, name="list_contextual_skills", description="Lists available high-level contextual skills (e.g., scientific-analysis, doc-coauthoring).")
        self.register(self.search_contextual_skills, name="search_contextual_skills", description="Searches for contextual skills by keyword.")
        self.register(self.load_skill_instructions, name="load_skill_instructions", description="Loads the detailed instructions/prompt for a specific contextual skill.")
        self.register(self.load_external_skill, name="load_external_skill", description="Loads instructions from a local SKILL.md file path.")

    def register(self, func: Callable, name: str = None, description: str = None):
        skill = Skill(func, name, description)
        self.skills[skill.name] = skill
        return skill

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self) -> List[Dict[str, Any]]:
        return [skill.to_tool_schema() for skill in self.skills.values()]

    def execute_tool(self, name: str, **kwargs):
        """
        Executes a tool by name with provided arguments.
        """
        if name in self.skills:
            return self.skills[name](**kwargs)
        raise ValueError(f"Skill {name} not found")

    def execute(self, name: str, **kwargs):
        return self.execute_tool(name, **kwargs)

    # --- Built-in Tools for Contextual Skills ---

    def list_contextual_skills(self) -> str:
        """
        Returns a formatted list of available contextual skills.
        """
        self.loader.refresh()
        skills = self.loader.list_skills()
        if not skills:
            return "No contextual skills found."
        
        output = ["Available Contextual Skills:"]
        for name, desc in skills.items():
            output.append(f"- {name}: {desc}")
        return "\n".join(output)

    def search_contextual_skills(self, query: str) -> str:
        """
        Searches for contextual skills by name or description.
        """
        self.loader.refresh()
        skills = self.loader.list_skills()
        matches = []
        query = query.lower()
        
        for name, desc in skills.items():
            if query in name.lower() or query in desc.lower():
                matches.append(f"- {name}: {desc}")
        
        if not matches:
            return f"No skills found matching '{query}'."
        
        return "Matching Skills:\n" + "\n".join(matches)

    def load_skill_instructions(self, skill_name: str) -> str:
        """
        Returns the full instructions for a given skill.
        The Agent should invoke this and then ingest the returned text into its context.
        """
        skill = self.loader.get_skill(skill_name)
        if skill:
            return f"--- INSTRUCTIONS FOR SKILL: {skill.name} ---\n{skill.instructions}\n--- END INSTRUCTIONS ---"
        return f"Error: Skill '{skill_name}' not found."
    def load_external_skill(self, path: str) -> str:
        p = path
        if os.path.isdir(p):
            f = os.path.join(p, "SKILL.md")
        else:
            f = p
        if not os.path.exists(f):
            return f"Error: Skill file not found at '{path}'."
        try:
            name, desc, instructions = self.loader._parse_skill_file(f)
            label = name if name else os.path.basename(os.path.dirname(f)) if os.path.dirname(f) else os.path.basename(f)
            return f"--- INSTRUCTIONS FOR SKILL: {label} ---\n{instructions}\n--- END INSTRUCTIONS ---"
        except Exception as e:
            return f"Error: Unable to load skill from '{path}': {e}"

# Global registry
registry = SkillRegistry()
