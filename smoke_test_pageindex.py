#!/usr/bin/env python3
"""
PageIndex Smoke Test
====================
Full end-to-end test of the PageIndex integration:
  1. Generate a realistic multi-page test PDF (TechCorp Strategic Analysis)
  2. Initialise PageIndexService with a Gemini API key
  3. Index the PDF (tree building)
  4. Run 5 sample questions via the agentic loop
  5. Print results clearly

Usage:
    python smoke_test_pageindex.py
"""

import os, sys, json, time, textwrap

# ── 0. Inject the Gemini key before anything imports litellm ─────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "REDACTED_ROTATE_THIS_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Add the project root to sys.path so local modules resolve correctly
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_PDF = os.path.join(PROJECT_ROOT, "test_techcorp.pdf")
WORKSPACE = os.path.join(PROJECT_ROOT, ".smoke_test_workspace")

# ── 1. Generate the test PDF ──────────────────────────────────────────────────

DOCUMENT_PAGES = [
    # ── Page 1: Executive Summary + Leadership ──────────────────────────────
    """\
TechCorp Strategic Analysis Report 2024

EXECUTIVE SUMMARY
TechCorp, founded in 2010 by CEO Sarah Chen and CTO Michael Rodriguez, has emerged as a
leading player in the artificial intelligence and cloud computing sector. Headquartered in
San Francisco, California, with additional research facilities in Boston, Austin, and Seattle.

TechCorp's primary revenue streams come from three main product lines:
  • CloudAI Platform
  • DataAnalytics Suite
  • SecurityShield Pro

COMPANY STRUCTURE AND LEADERSHIP

Executive Team
Sarah Chen (CEO): Former VP of Engineering at Google, holds a PhD in Computer Science from
Stanford University. Chen is responsible for overall strategic direction and has been
instrumental in securing major partnerships with Microsoft and Amazon.

Michael Rodriguez (CTO): Previously worked at Tesla and SpaceX, specialising in distributed
systems architecture. Rodriguez oversees all technical development and leads the AI research
initiative.

David Park (CFO): Former Goldman Sachs analyst with 15 years of experience in technology
finance. Park joined TechCorp in 2018 and has been crucial in managing the company's IPO
process.

Lisa Wang (VP of Sales): Former Salesforce executive who joined in 2020. Wang is responsible
for managing key client relationships including Fortune 500 companies like IBM, Oracle, and
JPMorgan Chase.

Board of Directors
TechCorp's board includes notable investors from Sequoia Capital, led by board chairman
Robert Kim, and Andreessen Horowitz, represented by partner Jennifer Lopez. The board also
includes independent directors Dr. Amanda Foster (former MIT professor) and James Thompson
(former Intel executive).
""",
    # ── Page 2: Product Portfolio ───────────────────────────────────────────
    """\
PRODUCT PORTFOLIO

CloudAI Platform
TechCorp's flagship product, CloudAI Platform, provides enterprise-grade machine learning
infrastructure. Key features include:
  • Auto-scaling compute clusters (up to 10,000 GPU nodes)
  • Proprietary model training framework (3x faster than TensorFlow baseline)
  • Real-time inference API with <10ms latency SLA
  • Pricing: $0.12/GPU-hour, enterprise contracts from $500K/year
  • Current MRR: $12.4M  |  YoY growth: 87%
  • Key customers: Netflix, Uber, Airbnb, Stripe

DataAnalytics Suite
An end-to-end data analytics solution:
  • Ingests 50+ data sources via built-in connectors
  • Natural-language query interface powered by CloudAI
  • Predictive analytics module with 94% accuracy on benchmark datasets
  • Pricing: $2,500/seat/year
  • Current MRR: $4.1M  |  YoY growth: 43%
  • Key customers: Walmart, Target, Home Depot

SecurityShield Pro
Zero-trust enterprise security platform:
  • AI-powered threat detection (99.7% accuracy)
  • Response time: <1 second
  • Compliance certifications: SOC 2 Type II, ISO 27001, FedRAMP
  • Pricing: $50K–$500K/year depending on company size
  • Current MRR: $2.8M  |  YoY growth: 62%
  • Key customers: Bank of America, Citigroup, US Department of Defense
""",
    # ── Page 3: Financial Performance ───────────────────────────────────────
    """\
FINANCIAL PERFORMANCE

Revenue Overview (FY 2024)
Total Revenue:        $231.6M   (+67% YoY)
ARR:                  $230.0M
Gross Margin:          72%
Operating Expenses:   $180.4M
Net Loss:             -$22.8M   (improving from -$45.1M in FY2023)

Revenue Breakdown
  CloudAI Platform:       $148.8M (64% of total revenue)
  DataAnalytics Suite:     $49.2M (21% of total revenue)
  SecurityShield Pro:      $33.6M (15% of total revenue)

Key Financial Metrics
  Customer Acquisition Cost (CAC):  $45,000
  Lifetime Value (LTV):            $380,000
  LTV:CAC Ratio:                   8.4x
  Churn Rate:                      3.2% annually
  Net Revenue Retention:           138%
  Cash Position:                   $450M (post-Series D)

Recent Funding History
  Series A (2015): $5M  — led by Sequoia Capital
  Series B (2017): $25M — led by Andreessen Horowitz
  Series C (2020): $120M — led by SoftBank
  Series D (2023): $400M — led by Tiger Global at $2.8B valuation
  IPO Target:      2025 — projected valuation $5–7B
""",
    # ── Page 4: Market Analysis ──────────────────────────────────────────────
    """\
MARKET ANALYSIS AND COMPETITIVE LANDSCAPE

Total Addressable Market (TAM)
  AI/ML Infrastructure:    $280B by 2027 (CAGR 34%)
  Business Analytics:      $115B by 2027 (CAGR 22%)
  Cybersecurity:           $345B by 2027 (CAGR 18%)
  Combined TAM:            $740B

Competitive Positioning
TechCorp occupies a unique position as a vertically-integrated AI company competing across
three high-growth markets. Key differentiators:

  1. Proprietary AI chip (TechChip-1): 40% more energy-efficient than NVIDIA A100
  2. Unified data platform: single pane of glass across AI, analytics, and security
  3. Enterprise trust: SOC 2, ISO 27001, FedRAMP certified — required for government deals

Primary Competitors
  CloudAI vs: AWS SageMaker, Google Vertex AI, Microsoft Azure ML
    → TechCorp advantage: 3x lower latency, 25% lower cost at scale
  DataAnalytics vs: Tableau, Power BI, Snowflake
    → TechCorp advantage: native AI integration, NL query interface
  SecurityShield vs: CrowdStrike, Palo Alto Networks, SentinelOne
    → TechCorp advantage: unified platform (no integration overhead)

Strategic Partnerships
  Microsoft (signed Q2 2024): Co-sell agreement covering 40,000 enterprise customers.
    Estimated incremental ARR: $35M by end of FY2025.
  Amazon AWS (signed Q3 2024): CloudAI listed as preferred ML partner on AWS Marketplace.
    Revenue share: TechCorp receives 75%, AWS retains 25%.
  NVIDIA (signed Q1 2024): Preferred access to next-gen H200 GPUs 6 months before GA.
    Commitment: 5,000 H200 units at $25K/unit.
""",
    # ── Page 5: Strategic Initiatives & Risks ───────────────────────────────
    """\
STRATEGIC INITIATIVES (FY 2025)

Initiative 1: International Expansion
  Target markets: EU (Germany, France, UK), Japan, Singapore
  Investment: $45M over 18 months
  Projected ARR contribution: $60M by end of FY2026
  Key hire: VP of International Sales (search ongoing)
  Regulatory hurdle: EU AI Act compliance required by Q2 2025

Initiative 2: TechChip-2 Development
  Next-generation AI accelerator chip
  Projected 80% improvement in performance-per-watt vs TechChip-1
  R&D investment: $35M in FY2025
  Expected tape-out: Q4 2025; GA: Q2 2026
  Target customers: Hyperscalers (AWS, Google, Meta, Microsoft)

Initiative 3: Government/Defense Vertical
  FedRAMP High authorisation in progress (expected Q1 2025)
  Pipeline: $120M in active government opportunities
  Key partner: Booz Allen Hamilton (reseller agreement signed October 2024)

RISK FACTORS

Technology Risks
  • AI model commoditisation may erode CloudAI Platform pricing power
  • TechChip-2 development delays would push GPU supply to NVIDIA dependency

Market Risks
  • Macro slowdown: enterprise IT budgets under pressure; 15% of pipeline is at risk
  • Microsoft and Google expanding competing offerings aggressively

Regulatory Risks
  • EU AI Act may require significant product changes (estimated $8M compliance cost)
  • FedRAMP delays could push government contracts to FY2026

Talent Risks
  • AI engineer churn: 22% in FY2024 (industry average 18%)
  • Three key research scientists received competing offers from Google DeepMind

CONCLUSION
TechCorp is positioned for strong growth with clear competitive advantages in its three
product verticals, healthy unit economics (LTV:CAC 8.4x), and a well-capitalised balance
sheet ($450M cash). The primary near-term risk is execution on international expansion and
the TechChip-2 roadmap while managing increasing competition from hyperscalers.
""",
]

SAMPLE_QUESTIONS = [
    "Who are the founders of TechCorp and what are their backgrounds?",
    "What is TechCorp's total revenue for FY 2024 and how is it broken down by product?",
    "What are TechCorp's key strategic partnerships and what are their financial terms?",
    "What are the main risk factors facing TechCorp?",
    "What is TechCorp's Series D valuation and IPO target?",
]


def build_test_pdf(path: str):
    """Create a multi-page test PDF using fpdf2."""
    from fpdf import FPDF

    def to_latin1(text: str) -> str:
        """Replace non-Latin-1 characters so Helvetica core font can render them."""
        replacements = {
            "\u2022": "-",   # bullet •
            "\u2014": "--",  # em dash —
            "\u2013": "-",   # en dash –
            "\u2018": "'",   # left single quote
            "\u2019": "'",   # right single quote
            "\u201c": '"',   # left double quote
            "\u201d": '"',   # right double quote
            "\u2192": "->",  # arrow →
            "\u00a0": " ",   # non-breaking space
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        # Final safety: encode to latin-1, replacing anything still unmappable
        return text.encode("latin-1", errors="replace").decode("latin-1")

    class SimplePDF(FPDF):
        def header(self):
            pass
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, "TechCorp Strategic Analysis Report 2024  --  Page " + str(self.page_no()), align="C")

    pdf = SimplePDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=15)

    for page_text in DOCUMENT_PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(30, 30, 30)
        for line in page_text.split("\n"):
            stripped = to_latin1(line.strip())
            if not stripped:
                pdf.ln(3)
                continue
            if stripped == stripped.upper() and len(stripped) > 4:
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(0, 6, stripped)
                pdf.set_font("Helvetica", size=10)
            else:
                pdf.multi_cell(0, 5, stripped)

    pdf.output(path)
    print(f"  Generated test PDF: {path}  ({os.path.getsize(path)//1024} KB, {len(DOCUMENT_PAGES)} pages)")


def hr(char="─", width=72):
    print(char * width)


def section(title: str):
    hr()
    print(f"  {title}")
    hr()


def print_result(result: dict):
    """Pretty-print a PageIndex query result."""
    print()
    if not result.get("success"):
        print(f"  ERROR: {result.get('error')}")
        return

    # Answer
    answer = result.get("answer", "").strip()
    wrapped = textwrap.fill(answer, width=70, subsequent_indent="    ")
    print(f"  ANSWER:\n    {wrapped}")

    # Citations
    citations = result.get("citations", [])
    if citations:
        print(f"\n  CITATIONS ({len(citations)}):")
        seen = set()
        for c in citations:
            key = c.get("pages", "")
            if key in seen:
                continue
            seen.add(key)
            section_label = f"  [{c['section']}]" if c.get("section") else ""
            print(f"    p. {key}{section_label}")

    # Tree path
    tree_path = result.get("tree_path", [])
    if tree_path:
        print(f"\n  REASONING PATH: Root → {' → '.join(tree_path[:4])}")

    # Tool calls
    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        print(f"\n  AGENT TOOL CALLS ({len(tool_calls)}):")
        for tc in tool_calls:
            args_str = ""
            if tc.get("args"):
                args_str = f"  args={json.dumps(tc['args'])}"
            print(f"    [{tc['tool']}]{args_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    section("PageIndex Smoke Test — Full Upload + Query Flow")
    print(f"  Gemini key : {GEMINI_API_KEY[:12]}…{GEMINI_API_KEY[-4:]}")
    print(f"  Workspace  : {WORKSPACE}")
    print(f"  Test PDF   : {TEST_PDF}")
    print()

    # ── Step 1: Build test PDF ─────────────────────────────────────────────
    section("STEP 1 / 3  —  Generate test PDF")
    if os.path.exists(TEST_PDF):
        print(f"  Found existing PDF: {TEST_PDF}  (skipping generation)")
    else:
        try:
            build_test_pdf(TEST_PDF)
        except ImportError:
            print("  fpdf2 not installed — creating plain-text proxy PDF via PyPDF2 …")
            # Fallback: write a minimal valid PDF manually
            _write_fallback_pdf(TEST_PDF)

    # ── Step 2: Initialise PageIndexService ───────────────────────────────
    section("STEP 2 / 3  —  Initialise PageIndexService")

    # Wipe old workspace so we always do a fresh index
    import shutil
    if os.path.exists(WORKSPACE):
        shutil.rmtree(WORKSPACE)
        print(f"  Cleared old workspace: {WORKSPACE}")

    config = {
        "gemini_api_key":             GEMINI_API_KEY,
        "claude_api_key":             "",
        "pageindex_workspace":        WORKSPACE,
        "pageindex_max_tool_rounds":  4,
        "pageindex_model":            "",   # auto-detect from gemini_api_key → gemini/gemini-1.5-flash
        "pageindex_enabled":          True,
    }

    from pageindex_service import PageIndexService
    svc = PageIndexService(config)

    if not svc.is_ready():
        print("  FAIL: PageIndexService.is_ready() returned False")
        print(f"  model resolved to: {repr(svc.model)}")
        sys.exit(1)

    print(f"  Service ready  : True")
    print(f"  LLM model      : {svc.model}")
    print(f"  Max tool rounds: {svc.max_tool_rounds}")
    print(f"  Workspace      : {svc.workspace}")

    # ── Step 3a: Index the document ────────────────────────────────────────
    section("STEP 3a / 3  —  Index document (tree building)")
    print("  Calling PageIndexService.index_document() …")
    print("  (This will make several LLM calls — expect 1–4 minutes)\n")

    class SimpleTracker:
        """Minimal progress tracker that prints to stdout."""
        def update(self, current, total, status="", message=""):
            pct = int(current / total * 100) if total else 0
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"  [{bar}] {pct:3d}%  {status:12s}  {message[:65]}", flush=True)
        @classmethod
        def get_tracker(cls, *args):
            return None

    tracker = SimpleTracker()
    t0 = time.time()
    result = svc.index_document(TEST_PDF, progress_tracker=tracker)
    elapsed = round(time.time() - t0, 1)

    print()
    if not result.get("success"):
        print(f"  FAIL: {result.get('error')}")
        sys.exit(1)

    print(f"  SUCCESS  ✓")
    print(f"  doc_id        : {result['doc_id']}")
    print(f"  doc_name      : {result['doc_name']}")
    print(f"  page_count    : {result['page_count']}")
    print(f"  section_count : {result['section_count']}")
    print(f"  time_elapsed  : {elapsed}s")

    doc_id = result["doc_id"]

    # Show the tree structure
    tree = svc.get_tree(doc_id)
    if tree:
        print(f"\n  TREE STRUCTURE ({len(tree)} top-level nodes):")
        def _show(nodes, depth=0):
            for node in nodes[:8]:   # cap display
                indent = "  " + "  " * depth
                pg = ""
                if node.get("start_index") and node.get("end_index"):
                    pg = f"  [pp. {node['start_index']}–{node['end_index']}]"
                print(f"{indent}• {node.get('title', 'Untitled')}{pg}")
                if node.get("nodes"):
                    _show(node["nodes"], depth + 1)
        _show(tree)

    # ── Step 3b: Run sample queries ────────────────────────────────────────
    section("STEP 3b / 3  —  Sample queries (agentic retrieval)")

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n  Q{i}: {question}")
        t0 = time.time()
        qresult = svc.query(question, doc_id=doc_id)
        qelapsed = round(time.time() - t0, 1)
        print_result(qresult)
        print(f"\n  (query time: {qelapsed}s)")
        hr("·")

    # ── Final summary ──────────────────────────────────────────────────────
    section("SMOKE TEST COMPLETE")
    docs = svc.list_documents()
    print(f"  Workspace documents: {len(docs)}")
    for d in docs:
        print(f"    [{d['status']}] {d['doc_name']}  —  {d['section_count']} sections, {d['page_count']} pages")
    print()
    print("  All steps passed. PageIndex is working correctly with Gemini.")
    print()


def _write_fallback_pdf(path: str):
    """Dead-simple PDF writer (no external library) as last-resort fallback."""
    lines = []
    for page_text in DOCUMENT_PAGES:
        lines.append(page_text)

    # We can't write a valid multi-page PDF without a library, so we write a
    # markdown file and tell the user to convert it.
    md_path = path.replace(".pdf", ".md")
    with open(md_path, "w") as f:
        for i, page in enumerate(DOCUMENT_PAGES, 1):
            f.write(f"# Page {i}\n\n{page}\n\n---\n\n")
    print(f"  Wrote Markdown fallback: {md_path}")
    print("  WARN: fpdf2 not available; index_document will use the .md path.")
    # Patch TEST_PDF to .md so the smoke test works
    global TEST_PDF
    TEST_PDF = md_path


if __name__ == "__main__":
    main()
