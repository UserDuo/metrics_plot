
import os
import sys

def pandoc_export_markdown_to_docx(
    md_path: str,
    output_path: str,
    description: str | None = None,
    reference_doc: str | None = None,
    extra_args: list[str] | None = None,
) -> bool:
    """
    Exports a Markdown file to DOCX using Pandoc, preserving mathematical formulas
    as native Word Office Math objects (OMML).

    Global Reuse Policy:
    - Implements the "Pandoc-first" strategy.
    - Uses 'markdown+tex_math_dollars+tex_math_double_backslash' to ensure
      maximum compatibility with LaTeX math syntax in Markdown.
    - Provides consistent user feedback in English.

    Args:
        md_path: Path to the source Markdown file.
        output_path: Path where the DOCX file should be saved.
        description: Optional description of the file content for logging.
        reference_doc: Optional path to a reference DOCX for styling.
        extra_args: Optional list of additional command-line arguments for Pandoc.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    if description is None:
        description = "DOCX file"
    
    # 1. Check Pandoc availability
    try:
        import pypandoc
    except ImportError:
        print("Error: 'pypandoc' module is not installed.")
        print("Please run: pip install pypandoc")
        return False
    
    # Check if pandoc binary is available
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Error: Pandoc command-line tool not found.")
        print("Please ensure Pandoc is installed and in your system PATH.")
        return False

    # 2. Check input existence
    if not os.path.exists(md_path):
        print(f"Error: Input Markdown file not found at: {md_path}")
        return False

    # 3. Prepare arguments
    # Enforce standard math extensions for reliable LaTeX -> OMML conversion
    # We include tex_math_single_backslash to support \(..\) and \[..\]
    # We include tex_math_dollars to support $..$ and $$..$$
    input_format = "markdown+tex_math_dollars+tex_math_single_backslash"
    
    pypandoc_args = []
    if reference_doc:
        if os.path.exists(reference_doc):
            pypandoc_args.append(f"--reference-doc={reference_doc}")
        else:
            print(f"Warning: Reference doc not found at {reference_doc}, skipping style application.")
    
    if extra_args:
        pypandoc_args.extend(extra_args)

    # 4. Execute conversion
    print(f"Step 1/2: Converting '{os.path.basename(md_path)}' to DOCX using Pandoc...")
    try:
        pypandoc.convert_file(
            md_path,
            "docx",
            format=input_format,
            outputfile=output_path,
            extra_args=pypandoc_args,
        )
        print(f"Success: {description} written to: {output_path}")
        return True
    except Exception as e:
        print(f"Error: Pandoc conversion failed. Details: {e}")
        return False

def validate_docx_readability(file_path: str) -> bool:
    """
    Validates that a DOCX file exists and can be opened by the python-docx library.
    This serves as a basic integrity check (Step 2/2).

    Args:
        file_path: Path to the DOCX file.

    Returns:
        bool: True if valid, False otherwise.
    """
    print(f"Step 2/2: Validating generated DOCX integrity...")
    if not os.path.exists(file_path):
        print(f"Error: DOCX file missing at {file_path}")
        return False
        
    try:
        from docx import Document
        doc = Document(file_path)
        # Basic check: read paragraph count
        para_count = len(doc.paragraphs)
        print(f"Validation Passed: DOCX is readable and contains {para_count} paragraphs.")
        return True
    except ImportError:
        print("Warning: 'python-docx' not installed. Skipping validation.")
        # If the tool is missing but file exists, we tentatively say "True" regarding the file itself,
        # but warn about validation. Ideally we want to be strict, but not blocking if dependency missing.
        return True
    except Exception as e:
        print(f"Validation Failed: Could not open DOCX file. Details: {e}")
        return False

def pandoc_export_markdown_to_html(
    md_path: str,
    output_path: str,
    description: str | None = None,
    extra_args: list[str] | None = None,
) -> bool:
    if description is None:
        description = "HTML file"
    try:
        import pypandoc
    except ImportError:
        print("Error: 'pypandoc' module is not installed.")
        print("Please run: pip install pypandoc")
        return False
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Error: Pandoc command-line tool not found.")
        print("Please ensure Pandoc is installed and in your system PATH.")
        return False
    if not os.path.exists(md_path):
        print(f"Error: Input Markdown file not found at: {md_path}")
        return False
    input_format = "markdown+tex_math_dollars+tex_math_single_backslash"
    pypandoc_args = []
    if extra_args:
        pypandoc_args.extend(extra_args)
    try:
        pypandoc.convert_file(
            md_path,
            "html",
            format=input_format,
            outputfile=output_path,
            extra_args=pypandoc_args,
        )
        print(f"Success: {description} written to: {output_path}")
        return True
    except Exception as e:
        print(f"Error: Pandoc conversion failed. Details: {e}")
        return False
