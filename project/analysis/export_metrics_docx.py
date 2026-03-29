import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.utils.docx_export import pandoc_export_markdown_to_docx, validate_docx_readability

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "results")
    md_path = os.path.join(results_dir, "CHAR_PERTURBATION_METRICS_17DIM.md")
    output_path = os.path.join(results_dir, "CHAR_PERTURBATION_METRICS_17DIM_NHB.docx")
    
    print("=== Pipeline Started: Metrics Documentation Export ===")
    
    # Step 1: Export with Pandoc (generates OMML for formulas)
    success = pandoc_export_markdown_to_docx(
        md_path,
        output_path,
        description="NHB-style metrics appendix DOCX",
    )
    
    if not success:
        print(
            "Pandoc path is unavailable or conversion failed. "
            "No DOCX was generated. Please ensure that pandoc and pypandoc "
            "are installed and accessible, then rerun this script."
        )
        sys.exit(1)
    
    # Step 2: Validation (and potential fallback logic placeholder)
    valid = validate_docx_readability(output_path)
    
    if valid:
        print("\nPipeline finished successfully.")
        print("It is now safe to open this file in Word.")
        print("Note: Mathematical formulas have been converted to Word Office Math objects.")
    else:
        print("\nWarning: DOCX generated but validation failed.")



if __name__ == "__main__":
    main()

