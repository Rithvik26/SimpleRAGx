"""
Plug-and-play domain configuration for SimpleRAG.

Each domain defines its metadata schema, LLM extraction prompt, and query behavior.
Switch domains by setting active_domain in config. The vc_financial domain is
the default and is tuned for VC dealflow, pitch decks, term sheets, and financial reports.
"""

DOMAINS = {
    "vc_financial": {
        "name": "VC & Startup Financial",
        "description": "Venture capital, startup fundraising, pitch decks, term sheets, and financial documents",
        "metadata_schema": {
            "doc_type": {
                "type": "str",
                "values": ["pitch_deck", "term_sheet", "cap_table", "lp_memo", "market_report", "filing", "other"],
            },
            "company_name": {"type": "str"},
            "sector": {"type": "str"},
            "stage": {
                "type": "str",
                "values": ["pre_seed", "seed", "series_a", "series_b", "series_c_plus", "growth", "unknown"],
            },
            "arr_usd": {"type": "int"},
            "valuation_usd": {"type": "int"},
            "founding_year": {"type": "int"},
            "geography": {"type": "str"},
            "founders": {"type": "list"},
            "investors": {"type": "list"},
        },
        "extraction_prompt": (
            "You are extracting structured metadata from a VC/financial document.\n"
            "Return ONLY valid JSON. Use null for any field you cannot determine.\n\n"
            "Fields:\n"
            '- doc_type: one of ["pitch_deck","term_sheet","cap_table","lp_memo","market_report","filing","other"]\n'
            "- company_name: the startup or company name\n"
            "- sector: industry sector (e.g. fintech, healthtech, saas, deeptech, consumer, marketplace, climate)\n"
            '- stage: one of ["pre_seed","seed","series_a","series_b","series_c_plus","growth","unknown"]\n'
            "- arr_usd: annual recurring revenue in USD as integer (null if not found)\n"
            "- valuation_usd: pre-money valuation in USD as integer (null if not found)\n"
            "- founding_year: year founded as integer (null if not found)\n"
            "- geography: primary market/geography (e.g. US, Southeast Asia, Europe, India)\n"
            "- founders: list of founder full names (empty list if none found)\n"
            "- investors: list of existing investor names (empty list if none found)\n\n"
            "Document text:\n{text}\n\nReturn ONLY JSON:"
        ),
        "system_prompt_suffix": (
            "\n\nYou are analyzing VC and financial documents. "
            "Prioritize investment-relevant details: team quality, market size, "
            "traction metrics (ARR, MoM growth), valuation, and competitive positioning."
        ),
        "filterable_fields": ["doc_type", "sector", "stage", "geography", "founding_year"],
    },

    "legal": {
        "name": "Legal Documents",
        "description": "Contracts, NDAs, agreements, and legal documents",
        "metadata_schema": {
            "doc_type": {
                "type": "str",
                "values": ["contract", "nda", "mou", "sla", "employment", "ip_assignment", "other"],
            },
            "parties": {"type": "list"},
            "jurisdiction": {"type": "str"},
            "effective_date": {"type": "str"},
            "expiry_date": {"type": "str"},
            "governing_law": {"type": "str"},
        },
        "extraction_prompt": (
            "Extract structured metadata from this legal document. Return ONLY valid JSON.\n\n"
            "Fields:\n"
            '- doc_type: one of ["contract","nda","mou","sla","employment","ip_assignment","other"]\n'
            "- parties: list of party names involved\n"
            "- jurisdiction: legal jurisdiction\n"
            "- effective_date: effective date as YYYY-MM-DD (null if not found)\n"
            "- expiry_date: expiry date as YYYY-MM-DD (null if not found)\n"
            "- governing_law: governing law/state\n\n"
            "Document text:\n{text}\n\nReturn ONLY JSON:"
        ),
        "system_prompt_suffix": (
            "\n\nYou are analyzing legal documents. "
            "Prioritize accuracy for parties, dates, obligations, and governing law."
        ),
        "filterable_fields": ["doc_type", "jurisdiction", "governing_law"],
    },

    "healthcare": {
        "name": "Healthcare & Life Sciences",
        "description": "Clinical documents, research papers, medical records, and pharma docs",
        "metadata_schema": {
            "doc_type": {
                "type": "str",
                "values": ["clinical_trial", "research_paper", "regulatory_filing", "protocol", "report", "other"],
            },
            "therapeutic_area": {"type": "str"},
            "phase": {"type": "str"},
            "institution": {"type": "str"},
            "indication": {"type": "str"},
            "authors": {"type": "list"},
        },
        "extraction_prompt": (
            "Extract structured metadata from this healthcare/life sciences document. Return ONLY valid JSON.\n\n"
            "Fields:\n"
            '- doc_type: one of ["clinical_trial","research_paper","regulatory_filing","protocol","report","other"]\n'
            "- therapeutic_area: e.g. oncology, cardiology, neurology\n"
            '- phase: clinical trial phase (e.g. "Phase 1", "Phase 2", "Phase 3") or null\n'
            "- institution: primary institution or company\n"
            "- indication: disease or condition being studied\n"
            "- authors: list of author names (empty list if none)\n\n"
            "Document text:\n{text}\n\nReturn ONLY JSON:"
        ),
        "system_prompt_suffix": (
            "\n\nYou are analyzing healthcare and life sciences documents. "
            "Prioritize clinical accuracy, safety data, efficacy metrics, and regulatory context."
        ),
        "filterable_fields": ["doc_type", "therapeutic_area", "phase", "indication"],
    },

    "general": {
        "name": "General Purpose",
        "description": "General purpose document QA — no domain-specific metadata",
        "metadata_schema": {
            "doc_type": {"type": "str"},
            "topic": {"type": "str"},
        },
        "extraction_prompt": (
            "Extract basic metadata from this document. Return ONLY valid JSON.\n\n"
            "Fields:\n"
            "- doc_type: document type (e.g. report, article, manual, presentation, other)\n"
            "- topic: main topic in 2-5 words\n\n"
            "Document text:\n{text}\n\nReturn ONLY JSON:"
        ),
        "system_prompt_suffix": "",
        "filterable_fields": ["doc_type", "topic"],
    },
}


def get_domain(domain_name: str) -> dict:
    """Return domain config by name, falling back to general."""
    return DOMAINS.get(domain_name, DOMAINS["general"])


def list_domains() -> list:
    """Return list of available domain names and descriptions."""
    return [{"name": k, "label": v["name"], "description": v["description"]} for k, v in DOMAINS.items()]
