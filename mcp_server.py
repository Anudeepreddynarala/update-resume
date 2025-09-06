from __future__ import annotations

import os
from typing import Any, Dict

from mcp.server import FastMCP


server = FastMCP(
    name="resume-optimizer-mcp",
    instructions=(
        "Optimize a LaTeX resume for a given job URL by updating Summary and Skills, "
        "then optionally compiling to PDF. Provide a job_url and an existing LaTeX "
        "resume filename from the latex-templates/ directory."
    ),
    tools=[],
)


@server.tool(
    name="optimize_resume",
    title="Optimize LaTeX resume for a job URL",
    description=(
        "Fetches the job description, analyzes it with Gemini, updates Summary and Skills "
        "in the provided LaTeX resume, writes a new .tex in output-pdfs/, and optionally compiles to PDF."
    ),
)
def optimize_resume_tool(
    job_url: str,
    resume_filename: str = "Anudeep_Narala.tex",
    compile_pdf: bool = True,
) -> Dict[str, Any]:
    """Optimize a LaTeX resume and return output file paths.

    Args:
        job_url: Public URL containing the job description to optimize for.
        resume_filename: LaTeX resume filename located under latex-templates/.
        compile_pdf: If true, compile the generated .tex to PDF using pdflatex.

    Returns:
        Dictionary containing company_name, job_title, output_tex_path, output_pdf_path (if compiled).
    """

    # Import project functions lazily to avoid import side-effects at server load
    from main import (
        extract_sections_from_resume,
        get_content_from_url,
        get_ai_optimizations,
        update_latex_resume,
        compile_latex_to_pdf,
    )

    resume_path = os.path.join("latex-templates", resume_filename)
    if not os.path.exists(resume_path):
        raise ValueError(f"Resume file not found: {resume_path}")

    with open(resume_path, "r") as f:
        resume_content = f.read()

    sections = extract_sections_from_resume(resume_content)

    job_description = get_content_from_url(job_url)
    if not job_description:
        raise ValueError("Failed to fetch job description from URL")

    optimizations = get_ai_optimizations(
        job_description,
        sections["summary"],
        sections["skills"],
    )
    if not optimizations:
        raise ValueError("Failed to get AI optimizations. Ensure GEMINI_API_KEY is set and valid.")

    company_name = optimizations.get("company_name", "Unknown_Company")
    job_title = optimizations.get("job_title", "Unknown_Job")

    output_dir = "output-pdfs"
    new_tex_path = update_latex_resume(
        resume_content,
        optimizations,
        company_name,
        job_title,
        output_dir,
    )

    if not new_tex_path:
        raise RuntimeError("LaTeX resume update failed; no output path returned.")

    output_pdf_path = new_tex_path.replace(".tex", ".pdf")

    if compile_pdf:
        compile_latex_to_pdf(new_tex_path)

    return {
        "company_name": company_name,
        "job_title": job_title,
        "output_tex_path": os.path.abspath(new_tex_path),
        "output_pdf_path": os.path.abspath(output_pdf_path) if compile_pdf else None,
    }


if __name__ == "__main__":
    server.run(transport="stdio")


