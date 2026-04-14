"""Generate the SimpleRAGx benchmark report PDF."""
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "simpleragx_benchmark_report.pdf")

# ── Colour palette ────────────────────────────────────────────────────────────
DARK  = colors.HexColor("#1a1a2e")
MID   = colors.HexColor("#2d4059")
LIGHT = colors.HexColor("#e8f4fd")
PASS  = colors.HexColor("#d4edda")
WARN  = colors.HexColor("#fff3cd")
FAIL  = colors.HexColor("#f8d7da")
ROW1  = colors.HexColor("#f8f9fa")
GREY  = colors.HexColor("#888888")

# ── Text styles ───────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_s    = S("T",  fontSize=20, leading=26, spaceAfter=6,  alignment=TA_CENTER, fontName="Helvetica-Bold",   textColor=DARK)
sub_s      = S("Su", fontSize=10, leading=14, spaceAfter=16, alignment=TA_CENTER, fontName="Helvetica",        textColor=GREY)
h1_s       = S("H1", fontSize=14, leading=18, spaceBefore=18, spaceAfter=6,  fontName="Helvetica-Bold",   textColor=DARK)
h2_s       = S("H2", fontSize=11, leading=15, spaceBefore=12, spaceAfter=4,  fontName="Helvetica-Bold",   textColor=MID)
body_s     = S("B",  fontSize=9,  leading=14, spaceAfter=4,  fontName="Helvetica",        textColor=colors.HexColor("#333333"))
small_s    = S("Sm", fontSize=8,  leading=12, spaceAfter=6,  fontName="Helvetica",        textColor=GREY)
caption_s  = S("Ca", fontSize=8,  leading=11, spaceAfter=4,  fontName="Helvetica-Oblique",textColor=GREY, alignment=TA_CENTER)

def sp(h=6):  return Spacer(1, h)
def hr():     return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dddddd"), spaceAfter=6, spaceBefore=6)
def p(text, style=None): return Paragraph(text, style or body_s)

def base_ts(nhead=1):
    return [
        ("BACKGROUND",   (0, 0),  (-1, nhead-1), MID),
        ("TEXTCOLOR",    (0, 0),  (-1, nhead-1), colors.white),
        ("FONTNAME",     (0, 0),  (-1, nhead-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0),  (-1, -1),       8.5),
        ("ALIGN",        (0, 0),  (-1, -1),       "LEFT"),
        ("VALIGN",       (0, 0),  (-1, -1),       "TOP"),
        ("ROWBACKGROUNDS",(0, nhead), (-1, -1),   [ROW1, colors.white]),
        ("GRID",         (0, 0),  (-1, -1),       0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING",  (0, 0),  (-1, -1),       6),
        ("RIGHTPADDING", (0, 0),  (-1, -1),       6),
        ("TOPPADDING",   (0, 0),  (-1, -1),       4),
        ("BOTTOMPADDING",(0, 0),  (-1, -1),       4),
    ]

def tbl(data, widths, extra_styles=None):
    ts = base_ts()
    if extra_styles:
        ts += extra_styles
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle(ts))
    return t

# ════════════════════════════════════════════════════════════════════════════
story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story += [
    sp(20),
    p("SimpleRAGx", S("Br", fontSize=11, fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
    p("RAG Mode Benchmark Report", title_s),
    p("TechCorp Road to IPO — Q1 2025  ·  GLiNER + Gemini 2.5 Flash  ·  April 2025", sub_s),
    hr(), sp(4),
]

# ── 1. Document Context ───────────────────────────────────────────────────────
story += [
    p("1. Document Under Test", h1_s),
    p("The benchmark was run against a simulated <b>pre-IPO investor relations document</b> for "
      "<b>TechCorp Inc.</b>, a fictional enterprise AI company modelled on real-world SaaS/AI "
      "growth-stage companies (referencing structures seen in OpenAI, Anthropic, Databricks, and "
      "Snowflake filings). The document was designed to stress-test each RAG mode across "
      "structured financial data, named-entity relationships, regulatory language, and strategic "
      "narrative — the exact content types that trip up production RAG systems."),
    sp(4),
    tbl([
        ["Field",         "Value"],
        ["Document",      "TechCorp Strategic Report: Road to IPO — Q1 2025"],
        ["Pages",         "5  (Executive Summary · Financials · Product/Leadership/M&A · Partnerships · Risk/IPO)"],
        ["Format",        "PDF — text-extractable, no scanned images"],
        ["Content types", "Financial tables, named entities, regulatory refs, strategic narrative, M&A details"],
        ["Entities",      "111 unique entities extracted  (people, orgs, products, financials, dates, locations)"],
        ["Relationships", "88 relationship triples extracted across 10 chunks"],
        ["Real-world refs","EU AI Act · NIST AI RMF · NASDAQ listing rules · Goldman Sachs · Morgan Stanley\n"
                           "NVIDIA GTC · Google Vertex AI · AWS ISV Accelerate · Microsoft Azure Marketplace"],
    ], [1.5*inch, 5.3*inch]),
    sp(6),
    p("<b>Key document facts the benchmark questions probe:</b>", h2_s),
]
for fact in [
    "Revenue: Q1 2025 total 72.4M (+86% YoY) — CloudAI 48.3M (67%), DataAnalytics 15.6M (22%), SecurityShield 8.5M (11%)",
    "Leadership: Sarah Chen (CEO, Stanford PhD, Google Brain / DeepMind), Michael Rodriguez (CTO, Tesla/SpaceX), David Park (CFO, ex-Stripe Wharton MBA)",
    "M&A: DataMesh Inc. acquired for 215M (140M cash + 75M stock); founders James Liu and Priya Venkataraman (ex-Snowflake, backed by Bessemer Ventures)",
    "Partnerships: Microsoft (80/20 Azure split, 38M TCV federal pipeline), AWS ISV Accelerate (12.3M ARR), Google Cloud (5M joint marketing + TPU v5), NVIDIA (B200 priority access)",
    "Risk factors: Technology (model commoditization by GPT-5/Gemini 2.5/Claude 4), Regulatory (EU AI Act, Bureau Veritas audit 3.2M), Competitive (Microsoft/Google/Databricks), Talent (22% churn)",
    "IPO: S-1 April 2025 → public June → roadshow July → NASDAQ:TCAI August 2025; Goldman Sachs projects 6.1B; management guidance 5.5B-7.2B",
    "Series D: Tiger Global, 400M at 2.8B valuation (September 2023)",
]:
    story.append(p("• " + fact))
story.append(sp(8))

# ── 2. Methodology ────────────────────────────────────────────────────────────
story += [
    hr(),
    p("2. Benchmark Methodology", h1_s),
    p("Each mode was given a fresh start — Qdrant collections, Neo4j graph, and PageIndex "
      "workspace were cleared before the run. All modes used <b>Gemini 2.5 Flash</b> for answer "
      "synthesis. Entity extraction used the new <b>GLiNER small-v2.1</b> hybrid pipeline "
      "(see Section 6). Five questions probe different retrieval challenges: biographical facts, "
      "structured financial data, multi-party relationship terms, categorical enumeration, and "
      "timeline/valuation reasoning."),
    sp(4),
    tbl([
        ["Component",                        "Tool / Model"],
        ["Answer synthesis (all modes)",     "gemini/gemini-2.5-flash via LiteLLM"],
        ["Entity extraction",                "GLiNER small-v2.1  (local model, zero API cost, ~0.2 s/chunk)*"],
        ["Relationship extraction",          "gemini/gemini-2.5-flash-lite via LiteLLM  (~2-4 s/chunk)"],
        ["Cypher generation (Neo4j mode)",   "gemini/gemini-2.5-flash-lite  (non-thinking, strict single-line output prompt)"],
        ["Embeddings",                       "Gemini text-embedding-004  (768-dim)"],
        ["Vector store",                     "Qdrant Cloud  (us-west-1)"],
        ["Graph database",                   "Neo4j Aura  (managed cloud)"],
        ["PageIndex workspace",              "/tmp/pi_bench_ws  — JSON hierarchical tree, no vector DB required"],
        ["Chunk size / overlap",             "1 000 chars / 200 chars overlap  →  10 chunks for the 5-page document"],
    ], [2.6*inch, 4.2*inch]),
    sp(4),
    p("* GLiNER (NAACL 2024, urchade/gliner_small-v2.1) is a 50M-param local NER model — no API calls, "
      "runs on CPU, ~0.2 s/chunk vs ~60 s/chunk with a thinking LLM. Used for entity spotting only; "
      "relationships are handled by the LLM stage.", small_s),
    sp(8),
]

# ── 3. Index Time ─────────────────────────────────────────────────────────────
story += [
    hr(),
    p("3. Index / Setup Time", h1_s),
    p("Time to fully index the 5-page document from scratch before any queries are run."),
    sp(4),
    tbl([
        ["Mode",                  "Time",    "What happens during indexing"],
        ["Normal RAG",            "5.0 s",   "Chunk text → embed (768-dim) → insert into Qdrant collection"],
        ["Graph RAG (Neo4j)",     "53.7 s",  "Chunk → GLiNER entities → flash-lite relationships → write to Neo4j Aura"],
        ["PageIndex",             "56.5 s",  "LLM reads full document → builds 5-section hierarchical TOC tree → writes JSON workspace"],
        ["Graph RAG (Vector)",    "230.5 s", "Chunk → GLiNER entities → flash-lite relationships → embed each entity → insert graph Qdrant collection"],
    ], [1.7*inch, 0.8*inch, 4.3*inch],
    extra_styles=[
        ("BACKGROUND", (0,1), (-1,1), PASS),
        ("BACKGROUND", (0,2), (-1,2), WARN),
        ("BACKGROUND", (0,3), (-1,3), WARN),
        ("BACKGROUND", (0,4), (-1,4), FAIL),
    ]),
    sp(4),
    p("Graph RAG Vector is slowest because it embeds every extracted entity individually and stores "
      "them in Qdrant on top of extraction. Neo4j skips the embedding step — entities write directly "
      "as structured graph properties.", small_s),
    sp(8),
]

# ── 4. Answer Quality ─────────────────────────────────────────────────────────
story += [
    hr(),
    p("4. Answer Quality Scorecard", h1_s),
    p("Each cell shows answer quality for the question/mode combination."),
    sp(4),
]

TICK = "\u2705"  # ✅
WARN_SYM = "\u26a0\ufe0f"  # ⚠️ — some fonts won't render emoji; use text instead
quality_data = [
    ["Question",                         "Normal RAG",              "Graph (Vector)",          "Graph (Neo4j)",             "PageIndex"],
    ["Q1: Founders +\nbackgrounds",      "OK - partial\n(bio short)","MISS - empty\n(0 results)","MISS - 0 rows\n(no Cypher match)","FULL\n+ page 3 cite"],
    ["Q2: Q1 2025 revenue\nbreakdown",   "OK\n(got it)",            "FULL\n+ breakdown + %",   "FULL\n15 rows",             "FULL\nall metrics"],
    ["Q3: Partnerships\n+ financial terms","FULL\nall 3 + terms",   "PARTIAL\nnames, no $",    "PARTIAL\nnames only",       "FULL\nterms + page 4"],
    ["Q4: Risk factors",                 "PARTIAL\ncompetitive only","PARTIAL\ncust. conc. only","MISS - 0 rows\n(no Cypher match)","FULL\nall 4 + page 5"],
    ["Q5: IPO timeline\n+ Series D",     "PARTIAL\ntimeline only",  "FULL\ntimeline + detail", "PARTIAL\nQ3 only",          "FULL\nGoldman 6.1B + Series D"],
    ["SCORE",                            "3 / 5",                   "2.5 / 5",                 "1.5 / 5",                   "5 / 5"],
]

cell_bg = [
    # row 1 (Q1)
    ("BACKGROUND",(1,1),( 1,1),WARN),("BACKGROUND",(2,1),(2,1),FAIL),("BACKGROUND",(3,1),(3,1),FAIL),("BACKGROUND",(4,1),(4,1),PASS),
    # row 2 (Q2)
    ("BACKGROUND",(1,2),( 1,2),WARN),("BACKGROUND",(2,2),(2,2),PASS),("BACKGROUND",(3,2),(3,2),PASS),("BACKGROUND",(4,2),(4,2),PASS),
    # row 3 (Q3)
    ("BACKGROUND",(1,3),( 1,3),PASS),("BACKGROUND",(2,3),(2,3),WARN),("BACKGROUND",(3,3),(3,3),WARN),("BACKGROUND",(4,3),(4,3),PASS),
    # row 4 (Q4)
    ("BACKGROUND",(1,4),( 1,4),WARN),("BACKGROUND",(2,4),(2,4),WARN),("BACKGROUND",(3,4),(3,4),FAIL),("BACKGROUND",(4,4),(4,4),PASS),
    # row 5 (Q5)
    ("BACKGROUND",(1,5),( 1,5),WARN),("BACKGROUND",(2,5),(2,5),PASS),("BACKGROUND",(3,5),(3,5),WARN),("BACKGROUND",(4,5),(4,5),PASS),
    # score row
    ("BACKGROUND",(0,6),(-1,6),colors.HexColor("#343a40")),
    ("TEXTCOLOR", (0,6),(-1,6),colors.white),
    ("FONTNAME",  (0,6),(-1,6),"Helvetica-Bold"),
    ("FONTSIZE",  (0,0),(-1,-1),7.5),
]

story += [
    tbl(quality_data, [1.5*inch, 1.12*inch, 1.12*inch, 1.12*inch, 1.12*inch], extra_styles=cell_bg),
    sp(4),
    p("PageIndex answered all 5 questions completely with page-level citations. Graph RAG (Vector) "
      "improved substantially with GLiNER — Q2 and Q5 moved from missed to full answers. Graph RAG "
      "(Neo4j) generates valid Cypher but the description fields in entity nodes don't always carry "
      "enough original text context for synthesis.", small_s),
    sp(8),
]

# ── 5. Query Time ─────────────────────────────────────────────────────────────
story += [
    hr(),
    p("5. Average Query Time  (5 questions each)", h1_s),
    sp(4),
    tbl([
        ["Mode",                  "Avg",    "Min",    "Max",    "Notes"],
        ["Graph RAG (Neo4j)",     "4.31 s", "3.15 s", "5.44 s", "Cypher exec is fast; bottleneck is LLM synthesis"],
        ["Graph RAG (Vector)",    "4.93 s", "4.33 s", "5.51 s", "Vector search + NetworkX traversal"],
        ["Normal RAG",            "5.17 s", "4.61 s", "5.84 s", "Embedding similarity + LLM synthesis"],
        ["PageIndex",             "7.16 s", "6.33 s", "8.32 s", "2-3 tool calls per query: structure -> page fetch -> synthesise"],
    ], [1.7*inch, 0.65*inch, 0.65*inch, 0.65*inch, 3.05*inch]),
    sp(8),
]

# ── 6. Post-Mortem ────────────────────────────────────────────────────────────
story += [hr(), p("6. Post-Mortem: Mode Analysis", h1_s)]

for name, analysis in [
    ("Normal RAG  —  3/5  —  Best all-rounder",
     "Indexed in 5 s. Retrieves the right text chunks for most questions via cosine similarity. "
     "Misses detail when a key fact spans a chunk boundary (risk factors only returned competitive "
     "risk, not all 4 categories). No infrastructure beyond a vector DB. Best default for "
     "general-purpose document Q&A."),
    ("Graph RAG (Vector)  —  2.5/5  —  Improved, still retrieval-limited",
     "With GLiNER, the extracted graph is now 3x richer (111 entities, 88 rels). Q2 revenue and Q5 "
     "IPO are now fully correct. Q1 founders still fails because the vector search for the question "
     "'who are the founders' returns financial entity nodes instead of person nodes — embedding "
     "similarity between question and entity description does not always map to the right entity "
     "type. Strongest for multi-document relationship traversal scenarios."),
    ("Graph RAG (Neo4j)  —  1.5/5  —  Fast queries, context-thin answers",
     "Cypher generation via gemini-2.5-flash-lite is now real and syntactically correct. Q2 (revenue, "
     "15 rows returned) and Q3 (partnerships, 8 rows) work well. Q1 and Q4 return 0 rows because "
     "the generated Cypher searched for relationship labels containing 'founder' or 'risk' — but "
     "GLiNER-extracted verbs are 'co-founded' and 'faces', which do contain those substrings but "
     "weren't matched due to exact string search semantics. Best value for persistent multi-document "
     "knowledge graphs queried repeatedly over time (org charts, legal networks, M&A tracking)."),
    ("PageIndex  —  5/5  —  Best quality, traceable, no vector DB",
     "Built a 5-section hierarchical tree in 56 s. Every question answered completely with exact "
     "page citations. The agent navigated directly to the right page for each question "
     "(revenue -> page 2, partnerships -> page 4, risks -> page 5). All financial figures exact, "
     "no hallucination. Slowest query time (7.16 s avg) because each question requires 2-3 LLM "
     "tool calls. Best mode for structured professional documents: annual reports, SEC filings, "
     "legal contracts, research papers, technical specifications."),
]:
    story += [p(name, h2_s), p(analysis), sp(4)]

story.append(sp(4))

# ── 7. Recommendation Matrix ──────────────────────────────────────────────────
# Use Paragraph in every cell so ReportLab wraps text instead of overflowing.
cell_s = ParagraphStyle("cell", fontSize=8.5, leading=12, fontName="Helvetica", textColor=colors.HexColor("#333333"))
hdr_s  = ParagraphStyle("hdr",  fontSize=8.5, leading=12, fontName="Helvetica-Bold", textColor=colors.white)

def row(use, mode, reason):
    return [Paragraph(use, cell_s), Paragraph(mode, cell_s), Paragraph(reason, cell_s)]

rec_rows = [
    [Paragraph("Use Case", hdr_s), Paragraph("Best Mode", hdr_s), Paragraph("Reason", hdr_s)],
    row("Quick Q&A on any pile of documents",              "Normal RAG",          "5 s index, any file type, solid answers"),
    row("Annual reports, SEC filings, financial docs",     "PageIndex",           "Page citations, exact numbers, structured navigation"),
    row("Exact figures, dates, regulatory clauses",        "PageIndex",           "Agent reads the right section directly"),
    row("Who worked with whom (no graph DB)",              "Graph RAG (Vector)",  "Entity graph + NetworkX multi-hop, only Qdrant needed"),
    row("Who worked with whom (persistent, multi-doc)",    "Graph RAG (Neo4j)",   "Survives restarts, graph grows richer with every doc"),
    row("Org charts, M&A networks, legal entity networks", "Graph RAG (Neo4j)",   "Cypher traversal across the full persistent graph"),
    row("Unknown document structure, quick setup",         "Normal RAG",          "Zero config, safe default choice"),
    row("Auditable answers with traceable page citations", "PageIndex",           "Every answer cites exact page numbers"),
]
story += [
    hr(),
    p("7. Recommendation Matrix", h1_s),
    sp(4),
    tbl(rec_rows, [2.5*inch, 1.55*inch, 2.75*inch]),
    sp(10),
    hr(),
    p("Generated by SimpleRAGx  \u00b7  April 2025", caption_s),
    p("Extractor: GLiNER small-v2.1 (NAACL 2024)  \u00b7  Synthesis: Gemini 2.5 Flash  "
      "\u00b7  Vector DB: Qdrant Cloud  \u00b7  Graph DB: Neo4j Aura", caption_s),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT, pagesize=letter,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
    topMargin=0.85*inch,  bottomMargin=0.85*inch,
)
doc.build(story)
print(f"Created: {OUTPUT}  ({os.path.getsize(OUTPUT):,} bytes)")
