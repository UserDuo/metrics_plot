
import os
import sys

# Ensure project is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.skills.registry import SkillRegistry

def main():
    print("Initializing SkillRegistry...")
    registry = SkillRegistry()
    
    print("\nRefreshing skills...")
    # Force refresh just in case
    registry.loader.refresh()
    
    print("\nListing Contextual Skills:")
    skills = registry.list_contextual_skills()
    print(skills)
    
    # Also verify internal list
    internal_list = registry.loader.list_skills()
    print(f"\nTotal skills found: {len(internal_list)}")
    
    expected_skills = [
        "algorithmic-art", "brand-guidelines", "canvas-design", "doc-coauthoring", 
        "docx", "frontend-design", "internal-comms", "mcp-builder", "pdf", 
        "pptx", "skill-creator", "slack-gif-creator", "theme-factory", 
        "web-artifacts-builder", "webapp-testing", "xlsx"
    ]
    
    missing = []
    found_names = [s['name'] for s in internal_list]
    for expected in expected_skills:
        if expected not in found_names:
            missing.append(expected)
            
    if missing:
        print(f"\nERROR: Missing skills: {missing}")
        sys.exit(1)
    else:
        print("\nSUCCESS: All expected skills are present.")

if __name__ == "__main__":
    main()
