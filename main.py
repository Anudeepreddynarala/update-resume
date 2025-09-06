import re
import subprocess
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import json
import sys

# Verbose logging control
VERBOSE = os.getenv("RESUME_OPTIMIZER_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# Helper function to escape LaTeX special characters
def escape_latex_chars(text: str) -> str:
    """
    Escapes special LaTeX characters in a string.
    """
    latex_special_chars = {
        '\\': r'\\textbackslash{}', # Re-enabled: Ensures all backslashes are escaped for LaTeX
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    for char, escape_seq in latex_special_chars.items():
        text = text.replace(char, escape_seq)
    return text

def get_content_from_url(url: str) -> str:
    try:
        # Use Jina AI Reader by prepending the URL
        jina_reader_url = f"https://r.jina.ai/{url}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(jina_reader_url, headers=headers, timeout=30) # Increased timeout
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL content with Jina AI Reader: {e.__class__.__name__}: {e}")
        print("This might be due to a network issue, invalid URL, or server-side problems.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching URL content with Jina AI Reader: {e.__class__.__name__}: {e}")
        print("This could be a parsing error or another unforeseen problem.")
        return None

def extract_sections_from_resume(resume_content: str) -> dict:
    summary = ""
    extracted_skills = {}
    
    # Manual parsing for summary section
    summary_start_marker = "\\section{Summary}"
    begin_onecolentry_marker = "\\begin{onecolentry}"
    end_onecolentry_marker = "\\end{onecolentry}"

    summary_section_start_idx = resume_content.find(summary_start_marker)
    if summary_section_start_idx != -1:
        # Look for \begin{onecolentry} after the summary section start
        begin_entry_idx = resume_content.find(begin_onecolentry_marker, summary_section_start_idx)
        if begin_entry_idx != -1:
            # Look for \end{onecolentry} after \begin{onecolentry}
            end_entry_idx = resume_content.find(end_onecolentry_marker, begin_entry_idx)
            if end_entry_idx != -1:
                # Extract the block between \begin{onecolentry} and \end{onecolentry}
                summary_block = resume_content[begin_entry_idx + len(begin_onecolentry_marker):end_entry_idx]
                
                # Now, use a simple regex to get content inside the first { } in the extracted block
                content_match = re.search(r"\s*\{([\s\S]*?)\}", summary_block)
                if content_match:
                    summary = content_match.group(1).strip()
                else:
                    vprint("Debug: Could not find content { } within the summary block.")
            else:
                vprint("Debug: Could not find \\end{onecolentry} after \\begin{onecolentry} for Summary.")
        else:
            vprint("Debug: Could not find \\begin{onecolentry} after \\section{Summary}.")
    else:
        vprint("Debug: Could not find \\section{Summary}.")

    # Manual parsing for skills section
    skills_section_start_marker = "\\section{Skills}"
    begin_onecolentry_marker = "\\begin{onecolentry}"
    end_onecolentry_marker = "\\end{onecolentry}"
    

    current_search_idx = resume_content.find(skills_section_start_marker)

    if current_search_idx != -1:
        vprint(f"Debug: Found \\section{{Skills}} at index {current_search_idx}.")
        # Move past the section marker
        current_search_idx += len(skills_section_start_marker)

        while True:
            begin_entry_idx = resume_content.find(begin_onecolentry_marker, current_search_idx)
            if begin_entry_idx == -1: # No more begin_onecolentry
                break
            
            end_entry_idx = resume_content.find(end_onecolentry_marker, begin_entry_idx)
            if end_entry_idx == -1: # Malformed, no matching end_onecolentry
                vprint("Debug: Found \\begin{onecolentry} but no matching \\end{onecolentry} in skills section.")
                break
            
            # Extract the content within the onecolentry block
            onecolentry_block_content = resume_content[begin_entry_idx + len(begin_onecolentry_marker) : end_entry_idx]
            
            # Now, use a simpler regex to extract category title and skills from this block
            # First, find the category title
            category_title_match = re.search(r"\\textbf{([^:]+):}", onecolentry_block_content)
            if category_title_match:
                category_title = category_title_match.group(1).strip()
                
                # Now, extract the skills list after the category title
                # Find the end of the category title (after the colon)
                title_end_idx = onecolentry_block_content.find(f"\\textbf{{{category_title}:}}") + len(f"\\textbf{{{category_title}:}}")
                
                # Extract the rest of the line as skills, stripping whitespace
                # This accounts for potential leading/trailing spaces around the actual skill list
                skill_list_text_raw = onecolentry_block_content[title_end_idx:].strip()
                
                # The skill_list_text_raw might contain \\n or \\vspace. We need to stop at the first true newline or \\vspace
                # Let's find the index of the first significant LaTeX command or newline after the skills
                # This is a bit tricky, but we can look for common delimiters
                newline_idx = skill_list_text_raw.find("\\n")
                vspace_idx = skill_list_text_raw.find("\\vspace")

                effective_end_idx = len(skill_list_text_raw)
                if newline_idx != -1: effective_end_idx = min(effective_end_idx, newline_idx)
                if vspace_idx != -1: effective_end_idx = min(effective_end_idx, vspace_idx)

                skill_list_text = skill_list_text_raw[:effective_end_idx].strip()

                extracted_skills[category_title] = skill_list_text
                vprint(f"Debug: Extracted Skill Category '{category_title}': {skill_list_text[:100]}...")
            else:
                vprint(f"Debug: Could not extract category title from block: {onecolentry_block_content[:100]}...")

            current_search_idx = end_entry_idx + len(end_onecolentry_marker)

    else:
        vprint("Debug: Could not find \\section{Skills}.")

    # We'll pass the extracted skills as a JSON string to the AI to maintain structure
    skills_json_string = json.dumps(extracted_skills)

    return {"summary": summary, "skills": skills_json_string}

# This function is no longer needed as AI optimizations handle skill extraction more comprehensively.
# def extract_skills_from_job_description(job_description: str) -> list:
#     # This is a very basic skill extraction. 
#     # In a real-world scenario, you'd use NLP libraries or a pre-trained model.
#     # For now, let's look for some common keywords.
#     common_skills = [
#         "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue",
#         "AWS", "Azure", "GCP", "Docker", "Kubernetes", "SQL", "NoSQL",
#         "Machine Learning", "Deep Learning", "Data Science", "TensorFlow", "PyTorch",
#         "Agile", "Scrum", "Project Management", "Communication", "Leadership"
#     ]
#     found_skills = []
#     job_description_lower = job_description.lower()
#     for skill in common_skills:
#         if skill.lower() in job_description_lower:
#             found_skills.append(skill)
#     return found_skills

    

def get_ai_optimizations(job_desc_text: str, current_summary: str, current_skills_section: str) -> dict:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please create a .env file with your API key.")
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    allowed_len = len(current_summary)

    prompt_template = (
        r"You are an expert career coach and LaTeX resume editor. Your task is to optimize a resume for a specific job description.\\n\\n" +
        r"**Job Description:**\\n" +
        r"---" +
        r"%s" +
        r"---" +
        r"**Current Resume Content:**\\n" +
        r"---" +
        r"Summary Section:\\n" +
        r"%s" + # current_summary_for_prompt
        r"Skill Categories Section:\\n" +
        r"%s" + # current_skills_json_string (will be implemented in next phase)
        r"---" +
        r"**Instructions:**\\n" +
        r"1. Revise the provided 'Summary' section to align with the role's requirements. Provide only the **plain text content** for the new summary. It **must not be empty and must contain concise and impactful text**, and its character count must be close to the original summary's content character count witin 5 characters (excluding LaTeX commands, %s characters). **It is critically important that the summary contains NO LaTeX commands or special characters; it should be pure plain text as it will be fully escaped before insertion.** Keep the summary to **two lines maximum** on a standard 10pt resume; as a hard cap keep it at or under **%s characters**. Focus the summary primarily on non-skill aspects from the job description (e.g., responsibilities, business impact, domain, outcomes), and include at most 1–2 high-priority skills from the job description. **Do not fabricate credentials, degrees, dates, or roles.** If the job title or description explicitly references a Product Owner role, mention the candidate is \"CSPO-certified\"; otherwise omit certification mentions.\\n" +
        r"2. Identify the top 10-15 most important hard skills, soft skills, and technologies mentioned in the job description.\\\\n" +
        r"3. Analyze *each* existing skill category in the 'Skills' section of the resume. For each category, create an optimized, comma-separated list of keywords. Integrate the most important skills from the job description while retaining the most relevant existing ones. Critically, remove any irrelevant skills to help ensure the resume fits on one page. Provide only the **plain text content** for the new skill lists. **Ensure each skill list is a non-empty, comma-separated string, and not null.**\\n" +
        r"   IMPORTANT: Collapse the skills into EXACTLY THREE categories and output ONLY these keys in 'updated_skill_categories': 'Management skills', 'Technical skills', and 'Tools'.\\n" +
        r"   - 'Management skills': product process, product/portfolio strategy, discovery, roadmapping, prioritization (RICE/MoSCoW), GTM, OKRs/KPIs, stakeholder mgmt, customer/market/domain terms (e.g., customer journey mapping, JTBD), communication/leadership.\\n" +
        r"   - 'Technical skills': AI/ML (LLMs, RAG), data/analytics/BI/experimentation (SQL, Tableau, Looker), programming/scripting, cloud/services, architecture concepts.\\n" +
        r"   - 'Tools': specific software/platforms (JIRA, Confluence, Figma, Postman, AWS, etc.).\\n" +
        r"   If a skill could fit multiple categories, choose the most natural fit; if still uncertain, default to 'Management skills'.\\n" +
        r"4. From the Job Description, extract the **Company Name** and **Job Title**. If you cannot find them explicitly, infer them from the context or use a placeholder like \"Unknown_Company\" or \"Unknown_Job\".\\n" +
        r"5. Your output **MUST** be a valid JSON object. Do not include any other text, explanations, or formatting like markdown backticks.\\n\\n" +
        r"**JSON Output Format:**\\n" +
        r"{\\n" +
        r"  \"updated_summary\": \"A highly skilled and driven professional with expertise in product management, poised to drive innovation at Hewlett Packard Enterprise.\",\\n" +
        r"  \"updated_skill_categories\": {\n" +
        r"    \"Management skills\": \"Roadmapping, Prioritization (RICE), Stakeholder Management, Customer Journey Mapping, OKRs/KPIs, GTM\",\\n" +
        r"    \"Technical skills\": \"SQL, Python, Tableau, LLMs, RAG, AWS\",\\n" +
        r"    \"Tools\": \"JIRA, Confluence, Figma, Postman, AWS, Excel\"\\n" +
        r"  },\\n" +
        r"  \"company_name\": \"Hewlett Packard Enterprise\",\\n" +
        r"  \"job_title\": \"Product Manager\"\\n" +
        r"}"
    )

    # Escape LaTeX special characters in inputs before sending to AI
    job_desc_text_for_prompt = job_desc_text.replace("\\", "\\\\")
    current_summary_for_prompt = current_summary.replace("\\", "\\\\")
    current_skills_section_for_prompt = current_skills_section.replace("\\", "\\\\")

    prompt = prompt_template % (
        job_desc_text_for_prompt,
        current_summary_for_prompt,
        current_skills_section_for_prompt, # JSON string
        allowed_len,
        allowed_len
    )

    attempts = 0
    last_error_note = ""
    while attempts < 3:
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                attempts += 1
                prompt += "\n\nRefinement: Return ONLY a valid JSON object per the specified schema."
                continue

            json_string_to_load = json_match.group(0)
            optimizations = json.loads(json_string_to_load)

            # Validate content
            if not optimizations or \
               not isinstance(optimizations.get("updated_summary"), str) or not optimizations.get("updated_summary") or \
               not isinstance(optimizations.get("company_name"), str) or not optimizations.get("company_name") or \
               not isinstance(optimizations.get("job_title"), str) or not optimizations.get("job_title") or \
               not isinstance(optimizations.get("updated_skill_categories"), dict) or not optimizations.get("updated_skill_categories"):
                attempts += 1
                prompt += "\n\nRefinement: The JSON was missing required keys or had empty values. Return all required fields with non-empty content."
                continue

            # Enforce hard summary cap
            new_summary = optimizations.get("updated_summary", "")
            if len(new_summary) > allowed_len + 5:
                attempts += 1
                over_by = len(new_summary) - (allowed_len + 5)
                prompt += f"\n\nRefinement: The previous summary exceeded the cap by {over_by} characters. Regenerate the summary under or equal to {allowed_len} characters total, maximum two lines, focusing on non-skill aspects and at most 1–2 skills."
                continue

            return optimizations
        except Exception as e:
            attempts += 1
            last_error_note = f"An error occurred during Gemini API call: {e.__class__.__name__}: {e}"
            prompt += "\n\nRefinement: Return ONLY a valid JSON object per the specified schema."
            continue

    # Fallback: return None so caller can decide, or truncate the summary safely
    print("Warning: Could not obtain compliant AI output after retries.")
    if 'optimizations' in locals():
        # Clip summary if necessary
        clipped = optimizations.get("updated_summary", "")[:allowed_len]
        optimizations["updated_summary"] = clipped
        return optimizations
    return None

def update_latex_resume(original_content: str, optimizations: dict, company_name: str, job_title: str, output_dir: str) -> str:
    updated_content = original_content

    # Update Summary (existing logic, no changes needed here for skills)
    new_plain_summary = optimizations.get("updated_summary")
    if new_plain_summary:
        # Aggressively clean LaTeX bolding and other commands from the plain summary
        # This regex removes \command{...} and \command patterns.
        new_plain_summary = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', new_plain_summary) # Removes \command{arg}
        new_plain_summary = re.sub(r'\\[a-zA-Z]+', '', new_plain_summary) # Removes \command
        new_plain_summary = re.sub(r'\s+', ' ', new_plain_summary).strip() # Normalize whitespace

        # Reconstruct the LaTeX summary block from plain text
        reconstructed_summary_block_content = escape_latex_chars(new_plain_summary)

        summary_start_marker = "\\section{Summary}"
        begin_onecolentry_marker = "\\begin{onecolentry}"
        end_onecolentry_marker = "\\end{onecolentry}"
        
        summary_section_start_idx = updated_content.find(summary_start_marker)
        if summary_section_start_idx != -1:
            begin_entry_idx = updated_content.find(begin_onecolentry_marker, summary_section_start_idx)
            if begin_entry_idx != -1:
                end_entry_idx = updated_content.find(end_onecolentry_marker, begin_entry_idx)
                if end_entry_idx != -1:
                    block_before_content = updated_content[begin_entry_idx : end_entry_idx]
                    relative_content_start_idx = block_before_content.find("{")
                    relative_content_end_idx = block_before_content.find("}", relative_content_start_idx + 1)

                    if relative_content_start_idx != -1 and relative_content_end_idx != -1:
                        # Calculate absolute indices for replacement
                        abs_content_start_idx = (begin_entry_idx + len(begin_onecolentry_marker)) + relative_content_start_idx + 1 # +1 to skip the '{'
                        abs_content_end_idx = (begin_entry_idx + len(begin_onecolentry_marker)) + relative_content_end_idx

                        # Debug prints for Summary Section Update
                        vprint("\n--- DEBUG: Summary Section Update ---")
                        vprint(f"New Plain Summary: {new_plain_summary[:100]}...")

                        summary_section_start_idx = updated_content.find(summary_start_marker)
                        if summary_section_start_idx != -1:
                            vprint("Found \\section{Summary}.")
                            begin_entry_idx = updated_content.find(begin_onecolentry_marker, summary_section_start_idx)
                            if begin_entry_idx != -1:
                                vprint("Found \\begin{onecolentry}.")
                                end_entry_idx = updated_content.find(end_onecolentry_marker, begin_entry_idx)
                                if end_entry_idx != -1:
                                    vprint("Found \\end{onecolentry}.")

                                    # Find the absolute indices of the opening and closing braces of the content
                                    content_start_abs_idx = updated_content.find("{", begin_entry_idx + len(begin_onecolentry_marker))
                                    content_end_abs_idx = updated_content.find("}", content_start_abs_idx + 1)

                                    if content_start_abs_idx != -1 and content_end_abs_idx != -1:
                                        # abs_content_start_idx should be *after* the opening '{'
                                        # abs_content_end_idx should be *before* the closing '}'
                                        abs_content_start_idx = content_start_abs_idx + 1
                                        abs_content_end_idx = content_end_abs_idx

                                        # Ensure the reconstructed_summary_block_content is wrapped in exactly one pair of braces
                                        # Extract the content from the reconstructed_summary_block_content if it already has braces
                                        temp_content_match = re.search(r'^\s*\{(.*)\}\s*$', reconstructed_summary_block_content, re.DOTALL)
                                        if temp_content_match:
                                            clean_content = temp_content_match.group(1).strip()
                                        else:
                                            clean_content = reconstructed_summary_block_content.strip()
                                        
                                        reconstructed_summary_block_content = f"{{{clean_content}}}"

                                        vprint(f"  summary_section_start_idx: {summary_section_start_idx}")
                                        vprint(f"  begin_entry_idx: {begin_entry_idx}")
                                        vprint(f"  end_entry_idx: {end_entry_idx}")
                                        vprint(f"  Content Start Abs Index (after {{'{{'}}): {abs_content_start_idx}")
                                        vprint(f"  Content End Abs Index (before {{'}}'}}): {abs_content_end_idx}")
                                        vprint(f"  Content to be replaced (from original): \n---{updated_content[abs_content_start_idx:abs_content_end_idx][:200]}---\n")
                                        vprint(f"  Part before replacement (first 200 chars): \n---{updated_content[:abs_content_start_idx][:200]}---\n")
                                        vprint(f"  Part after replacement (first 200 chars): \n---{updated_content[abs_content_end_idx:][:200]}---\n")

                                        # Perform the replacement using string slicing
                                        reconstructed_summary_full_block = (
                                            f"\\begin{{onecolentry}}\n"
                                            f"            {reconstructed_summary_block_content}\n"
                                            f"        \\end{{onecolentry}}"
                                        )
                                        updated_content = (
                                            updated_content[:begin_entry_idx] +
                                            reconstructed_summary_full_block +
                                            updated_content[end_entry_idx + len(end_onecolentry_marker):]
                                        )
                                        vprint("Summary content updated successfully.")
                                    else:
                                        vprint("Debug: Could not find content { } within the summary block for update.")
                                else:
                                    print("Warning: Could not find \\end{onecolentry} after \\begin{onecolentry} for Summary update.")
                            else:
                                print("Warning: Could not find \\begin{onecolentry} after \\section{Summary} for update.")
                        else:
                            print("Warning: Summary section structure not found for update.")
                        vprint("-------------------------------------")
                else:
                    vprint("Debug: Could not find content { } within the summary block for update.")
            else:
                print("Warning: Could not find \\begin{onecolentry} after \\section{Summary} for update.")
        else:
            print("Warning: Summary section structure not found for update.")
        vprint("-------------------------------------")

    # Update specific skill categories
    updated_skill_categories = optimizations.get("updated_skill_categories", {})
    vprint(f"\n--- DEBUG: Skills Section Update ---")
    vprint(f"AI Optimizations for Skills (raw): {optimizations.get('updated_skill_categories', 'Not Found')}")
    vprint(f"Parsed updated_skill_categories: {updated_skill_categories}")
    # Rebuild the entire Skills section using the new three categories
    skills_section_start_marker = "\\section{Skills}"
    begin_onecolentry_marker = "\\begin{onecolentry}"
    end_onecolentry_marker = "\\end{onecolentry}"

    # Normalize keys to lower for matching
    normalized_updated = { (k or "").strip().lower(): (v or "") for k, v in updated_skill_categories.items() }
    desired_order = [
        ("Management skills", "management skills"),
        ("Technical skills", "technical skills"),
        ("Tools", "tools"),
    ]

    def normalize_and_limit(category_display_name: str, skills_csv: str) -> str:
        items_raw = [s.strip() for s in (skills_csv or "").split(',') if s.strip()]
        seen = set()
        items = []
        for itm in items_raw:
            key = itm.lower()
            if key not in seen:
                seen.add(key)
                items.append(itm)
        max_items_map = {
            'Management skills': 12,
            'Technical skills': 12,
            'Tools': 14,
        }
        max_items = max_items_map.get(category_display_name, 12)
        return ", ".join(items[:max_items])

    rebuilt_blocks = []
    for display_name, norm_key in desired_order:
        skills_csv = normalized_updated.get(norm_key, "").strip()
        if not skills_csv:
            continue
        skills_csv = normalize_and_limit(display_name, skills_csv)
        if not skills_csv:
            continue
        block = (
            f"\\begin{{onecolentry}}\n"
            f"            \\textbf{{{display_name}:}} {escape_latex_chars(skills_csv)}\n"
            f"        \\end{{onecolentry}}"
        )
        rebuilt_blocks.append(block)

    skills_section_start_idx = updated_content.find(skills_section_start_marker)
    if skills_section_start_idx != -1 and rebuilt_blocks:
        # Find section end
        next_section_idx = updated_content.find("\\section{", skills_section_start_idx + len(skills_section_start_marker))
        end_document_idx = updated_content.find("\\end{document}", skills_section_start_idx + len(skills_section_start_marker))
        if next_section_idx != -1 and (end_document_idx == -1 or next_section_idx < end_document_idx):
            skills_section_end_idx = next_section_idx
        elif end_document_idx != -1:
            skills_section_end_idx = end_document_idx
        else:
            skills_section_end_idx = len(updated_content)

        # Locate where the first onecolentry begins (if any) after the heading
        first_begin_entry_abs_idx = updated_content.find(begin_onecolentry_marker, skills_section_start_idx)
        if first_begin_entry_abs_idx == -1 or first_begin_entry_abs_idx > skills_section_end_idx:
            # No existing blocks; insert after the section header line
            insertion_point = skills_section_start_idx + len(skills_section_start_marker)
            # Include any whitespace/newlines right after header
            header_to_end = updated_content[skills_section_start_idx:skills_section_end_idx]
            header_trailing_idx = header_to_end.find('\n')
            if header_trailing_idx != -1:
                insertion_point = skills_section_start_idx + header_trailing_idx + 1
            new_skills_section_content = "\n\n".join(rebuilt_blocks)
            updated_content = (
                updated_content[:insertion_point] +
                "\n\n" + new_skills_section_content + "\n" +
                updated_content[skills_section_end_idx:]
            )
        else:
            pre_skills_section_content = updated_content[:skills_section_start_idx]
            post_skills_section_content = updated_content[skills_section_end_idx:]
            section_heading_and_pre_onecolentry_whitespace = updated_content[skills_section_start_idx : first_begin_entry_abs_idx]
            new_skills_section_content = "\n\n".join(rebuilt_blocks)
            updated_content = (
                pre_skills_section_content +
                section_heading_and_pre_onecolentry_whitespace +
                "\n\n" +
                new_skills_section_content +
                post_skills_section_content
            )
        vprint("Debug: Skills section rebuilt to three categories.")
    else:
        vprint("Debug: Skills section not rebuilt (either not found or no categories provided).")
    vprint("-------------------------------------")

    # Save the new .tex file
    sanitized_company = re.sub(r'[^a-zA-Z0-9_]', '', company_name.replace(' ', '_'))
    sanitized_job_title = re.sub(r'[^a-zA-Z0-9_]', '', job_title.replace(' ', '_'))

    new_file_name = f"{sanitized_company}_{sanitized_job_title}.tex"
    new_file_path = os.path.join(output_dir, new_file_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(new_file_path, 'w') as new_file:
        new_file.write(updated_content)
    
    return new_file_path

def compile_latex_to_pdf(tex_file_path: str):
    if not tex_file_path:
        return
        
    output_directory = os.path.dirname(tex_file_path)

    command = [
        'pdflatex',
        '-interaction=nonstopmode', # Added to prevent interactive hangs on error
        '-output-directory', output_directory,
        tex_file_path
    ]
    
    try:
        subprocess.run(command, check=True)
        subprocess.run(command, check=True)
        
        pdf_path = tex_file_path.replace('.tex', '.pdf')
        print(f"\n✅ Success! Your new resume is ready: {pdf_path}")

    except FileNotFoundError:
        print("\n❌ Error: 'pdflatex' command not found.")
        print("Please make sure you have a full MacTeX installation and that it's in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred during PDF compilation: {e}")
        print("Check for errors in your .tex file or the LaTeX installation.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Modified main function to accept arguments directly or from command line
def main():
    # Check if arguments are passed via command line (e.g., from a tool call)
    if len(sys.argv) > 1:
        try:
            json_args = json.loads(sys.argv[1])
            job_url = json_args["job_url"]
            resume_file_name = json_args["resume_filename"]
            optimize_resume(job_url, resume_file_name)
        except json.JSONDecodeError:
            print("Error: Invalid JSON arguments provided.")
        except KeyError as e:
            print(f"Error: Missing expected argument: {e}")
    else:
        # Original interactive mode for direct script execution
        job_url = input("Please enter the job posting URL: ")
        resume_file_name = input("Please enter the name of your resume .tex file (e.g., Anudeep_Narala.tex): ")
        optimize_resume(job_url, resume_file_name)


def optimize_resume(job_url: str, resume_file_name: str):
    resume_file_path = os.path.join("latex-templates", resume_file_name)

    if not os.path.exists(resume_file_path):
        print(f"Error: Resume file not found at {resume_file_path}")
        return

    print(f"Processing job URL: {job_url}")
    print(f"Using resume file: {resume_file_path}")

    with open(resume_file_path, 'r') as f:
        resume_content = f.read()

    sections = extract_sections_from_resume(resume_content)
    print("--- Extracted Resume Sections ---")
    print(f"Summary: {sections['summary'][:200]}...")
    print(f"Skills: {sections['skills'][:200]}...")

    print("Fetching job description...")
    job_description = get_content_from_url(job_url)
    if not job_description:
        print("Failed to fetch job description. Exiting.")
        return
    
    print(f"Job Description (first 200 chars): {job_description[:200]}...")

    print("Getting AI optimizations for summary and skills...")
    optimizations = get_ai_optimizations(job_description, sections['summary'], sections['skills'])

    if not optimizations:
        print("Failed to get AI optimizations. Exiting.")
        return

    company_name = optimizations.get("company_name", "Unknown_Company")
    job_title = optimizations.get("job_title", "Unknown_Job")

    print(f"Optimizing resume for {job_title} at {company_name}...")
    
    output_dir = "output-pdfs"
    updated_tex_path = update_latex_resume(resume_content, optimizations, company_name, job_title, output_dir)
    
    if not updated_tex_path:
        print("Failed to update LaTeX resume. Exiting.")
        return

    print(f"Updated LaTeX file saved to: {updated_tex_path}")

    print("Compiling optimized resume to PDF...")
    compile_latex_to_pdf(updated_tex_path)

if __name__ == "__main__":
    main()