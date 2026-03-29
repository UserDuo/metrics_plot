import os
import yaml
from typing import Dict, Optional, Tuple

class Skill:
    def __init__(self, name: str, description: str, instructions: str, path: str):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.path = path

class SkillLoader:
    def __init__(self, library_path: str):
        self.library_path = library_path
        self.skills: Dict[str, Skill] = {}
        self.refresh()

    def refresh(self):
        """Scans the library path for skills."""
        self.skills = {}
        if not os.path.exists(self.library_path):
            return

        for item in os.listdir(self.library_path):
            item_path = os.path.join(self.library_path, item)
            skill_file = os.path.join(item_path, "SKILL.md")
            
            if os.path.isdir(item_path) and os.path.exists(skill_file):
                try:
                    name, desc, instructions = self._parse_skill_file(skill_file)
                    # Fallback name from directory if not in frontmatter
                    final_name = name if name else item
                    self.skills[final_name] = Skill(final_name, desc, instructions, item_path)
                except Exception as e:
                    print(f"Error loading skill {item}: {e}")

    def _parse_skill_file(self, file_path: str) -> Tuple[Optional[str], str, str]:
        """Parses a SKILL.md file with YAML frontmatter."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split frontmatter and body
        if content.startswith('---'):
            try:
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter_raw = parts[1]
                    body = parts[2].strip()
                    
                    metadata = yaml.safe_load(frontmatter_raw)
                    name = metadata.get('name')
                    description = metadata.get('description', '')
                    
                    return name, description, body
            except yaml.YAMLError:
                pass
        
        # Fallback if no valid frontmatter
        return None, "No description provided.", content

    def list_skills(self) -> Dict[str, str]:
        """Returns a dict of skill name -> description."""
        return {name: skill.description for name, skill in self.skills.items()}

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)
