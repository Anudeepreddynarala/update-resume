## Resume Optimizer

AI-powered CLI and MCP server to tailor your LaTeX resume to a job posting.

### Quick Start

1) Create venv and install deps

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Add API key

```bash
echo GEMINI_API_KEY=your_key_here > .env
```

3) Place your LaTeX resume in `latex-templates/` (e.g., `Anudeep_Narala.tex`).

4) Run

```bash
python main.py '{"job_url": "https://en.wikipedia.org/wiki/Engineering", "resume_filename": "Anudeep_Narala.tex"}'
```

Output `.tex`/`.pdf` files are written to `output-pdfs/`.

### Use as an MCP Tool

This repo includes `mcp_server.py` exposing tool `optimize_resume`.

- Claude Desktop

```bash
mcp install mcp_server.py --name resume-optimizer-mcp --env-file .env
```

Restart Claude; you’ll see server `resume-optimizer-mcp` with tool `optimize_resume`.

- Cursor

Project-level config: add `.cursor/mcp.json` (Cursor reads this file, not `cursor.json`).

Option A: Direct Python (uses your venv’s Python)

```json
{
  "mcpServers": {
    "resume-optimizer-mcp": {
      "command": "/Users/you/path/update-resume/venv/bin/python",
      "args": [
        "/Users/you/path/update-resume/mcp_server.py"
      ],
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

Option B: uv + mcp runner

```json
{
  "mcpServers": {
    "resume-optimizer-mcp": {
      "command": "uv",
      "args": ["run","--with","mcp[cli]","mcp","run","/Users/you/path/update-resume/mcp_server.py"],
      "env": {"GEMINI_API_KEY": "${GEMINI_API_KEY}"}
    }
  }
}
```

Then fully restart Cursor (quit and reopen). Tool appears under Integrations → MCP.

Call tool `optimize_resume` with:

```json
{
  "job_url": "https://en.wikipedia.org/wiki/Engineering",
  "resume_filename": "Anudeep_Narala.tex",
  "compile_pdf": true
}
```

### Notes

- Requires Python 3.9–3.11, MacTeX (`pdflatex`) for PDF, and a Gemini API key.
- Enable verbose debug logs by setting `RESUME_OPTIMIZER_VERBOSE=1`.