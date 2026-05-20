"""Extraction schemas for thermal-comfort literature reviews.

The 33 columns of the Excel template are modelled here as two Pydantic
classes:

* :class:`StructuredFields` covers the 27 factual fields extracted in the
  first LLM pass with strict JSON-schema validation.
* :class:`SynthesisFields` covers the 6 free-form fields produced in the
  second pass, where the LLM is asked to summarise rather than extract.

Splitting the schema in two lets us use the right prompting style for
each kind of field and keeps the JSON returned by the LLM small enough
to validate reliably.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StructuredFields(BaseModel):
    """Factual fields extracted directly from the article.

    Every field is optional at the model level: the LLM is instructed to
    return an empty string when a value is not stated in the paper, which
    is preferable to hallucinating a plausible-looking answer.
    """

    model_config = ConfigDict(extra="forbid")

    article_name: str = Field(
        default="",
        description="Full title of the article exactly as printed.",
    )
    key_words: str = Field(
        default="",
        description="Author-provided keywords, comma-separated.",
    )
    year: str = Field(
        default="",
        description="Year of publication as a 4-digit string.",
    )
    journal: str = Field(
        default="",
        description="Name of the journal or conference proceedings.",
    )
    researchers: str = Field(
        default="",
        description="Authors, comma-separated, in the order they appear.",
    )
    country: str = Field(
        default="",
        description="Country where the study was conducted.",
    )
    city: str = Field(
        default="",
        description="City where the study was conducted.",
    )
    climate: str = Field(
        default="",
        description=(
            "Köppen-Geiger climate classification if stated, otherwise a "
            "short description (e.g. 'hot-humid', 'temperate continental')."
        ),
    )
    subjects_of_study: str = Field(
        default="",
        description="Number and type of participants (e.g. '120 office workers').",
    )
    seasons_of_study: str = Field(
        default="",
        description="Season(s) during which measurements were taken.",
    )
    spaces_studied: str = Field(
        default="",
        description="Type of space studied (indoor office, outdoor plaza, etc.).",
    )
    age: str = Field(
        default="",
        description="Age range or mean age of participants.",
    )
    gender: str = Field(
        default="",
        description="Gender distribution of participants.",
    )
    ethnicity: str = Field(
        default="",
        description="Ethnic background of participants if reported.",
    )
    behaviours_activity: str = Field(
        default="",
        description="Metabolic rate / activity level (met value if given).",
    )
    clothing_level: str = Field(
        default="",
        description="Clothing insulation level (clo value if given).",
    )
    numerical_variables: str = Field(
        default="",
        description="Quantitative variables measured (Ta, RH, Tmrt, PMV, etc.).",
    )
    qualitative_variables: str = Field(
        default="",
        description="Qualitative variables collected (thermal sensation votes, etc.).",
    )
    questionnaire_extent: str = Field(
        default="",
        description="Size or scope of the questionnaire (number of items, scale).",
    )
    questionning_time: str = Field(
        default="",
        description="When and how often the questionnaire was administered.",
    )
    questionnaire_questions: str = Field(
        default="",
        description="Brief summary of the questions asked.",
    )
    kpi: str = Field(
        default="",
        description="Key performance indicators (PMV, PPD, UTCI, SET*, etc.).",
    )
    urban_cooling_strategies: str = Field(
        default="",
        description="Urban-scale cooling strategies investigated.",
    )
    personal_cooling_strategies: str = Field(
        default="",
        description="Personal-scale cooling strategies investigated.",
    )
    urban_heating_strategies: str = Field(
        default="",
        description="Urban-scale heating strategies investigated.",
    )
    personal_heating_strategies: str = Field(
        default="",
        description="Personal-scale heating strategies investigated.",
    )
    software_used: str = Field(
        default="",
        description=(
            "Software, simulation tools, or statistical packages used. "
            "State if a native/custom tool was developed."
        ),
    )


class SynthesisFields(BaseModel):
    """Summary fields the LLM writes by reading the whole article.

    Unlike :class:`StructuredFields`, these are not present verbatim in
    the source text; the LLM has to synthesise them.
    """

    model_config = ConfigDict(extra="forbid")

    research_questions: str = Field(
        default="",
        description="The research questions the article sets out to answer.",
    )
    key_goals: str = Field(
        default="",
        description="Stated goals or objectives of the study.",
    )
    methodology: str = Field(
        default="",
        description="Short paragraph summarising the methodology.",
    )
    notes: str = Field(
        default="",
        description="Notable details, limitations, or caveats.",
    )
    brief_double_click_to_see_all: str = Field(
        default="",
        description="One-paragraph brief covering scope and main findings.",
    )
    g_m_r_brief: str = Field(
        default="",
        description=(
            "Goals-Methodology-Results brief: a single paragraph that "
            "states the goals, the methodology, and the headline results."
        ),
    )


class ArticleExtraction(BaseModel):
    """Combined result of both extraction passes for one article."""

    structured: StructuredFields
    synthesis: SynthesisFields


# Ordered list of the 33 Excel column headers exactly as they appear in
# the template. The order is the column order in the output workbook.
# Pairing this with the Pydantic field names below means we can map
# model values to the correct Excel column without relying on field
# ordering inside Python (which would be fragile).
EXCEL_COLUMNS: tuple[str, ...] = (
    "Article Name",
    "Key Words",
    "Year",
    "Journal",
    "Researchers",
    "Country",
    "City",
    "Climate",
    "Subjects of Study",
    "Seasons of Study",
    "Spaces Studied",
    "Age",
    "Gender",
    "ethnicity",
    "Behaviours Activity",
    "Clothing Level",
    "Numerical Variables",
    "Qualitative Variables",
    "Questionnare Extent",
    "Questionning Time",
    "Questionnare Questions",
    "KPI",
    "Urban Cooling Strategies",
    "Personal Cooling Strategies",
    "Urban Heating Strategies",
    "Personal Heating Strategies",
    "Software Used (State if Native)",
    "Research Questions",
    "Key Goals",
    "Methodology",
    "Notes",
    "Brief Double Click to See All",
    "G-M-R Brief",
)


# Map each Excel column header to the matching model field name.
# Defined explicitly so renaming a Python field never silently breaks
# the Excel output.
COLUMN_TO_FIELD: dict[str, str] = {
    "Article Name": "article_name",
    "Key Words": "key_words",
    "Year": "year",
    "Journal": "journal",
    "Researchers": "researchers",
    "Country": "country",
    "City": "city",
    "Climate": "climate",
    "Subjects of Study": "subjects_of_study",
    "Seasons of Study": "seasons_of_study",
    "Spaces Studied": "spaces_studied",
    "Age": "age",
    "Gender": "gender",
    "ethnicity": "ethnicity",
    "Behaviours Activity": "behaviours_activity",
    "Clothing Level": "clothing_level",
    "Numerical Variables": "numerical_variables",
    "Qualitative Variables": "qualitative_variables",
    "Questionnare Extent": "questionnaire_extent",
    "Questionning Time": "questionning_time",
    "Questionnare Questions": "questionnaire_questions",
    "KPI": "kpi",
    "Urban Cooling Strategies": "urban_cooling_strategies",
    "Personal Cooling Strategies": "personal_cooling_strategies",
    "Urban Heating Strategies": "urban_heating_strategies",
    "Personal Heating Strategies": "personal_heating_strategies",
    "Software Used (State if Native)": "software_used",
    "Research Questions": "research_questions",
    "Key Goals": "key_goals",
    "Methodology": "methodology",
    "Notes": "notes",
    "Brief Double Click to See All": "brief_double_click_to_see_all",
    "G-M-R Brief": "g_m_r_brief",
}


def extraction_to_row(extraction: ArticleExtraction) -> list[str]:
    """Flatten an :class:`ArticleExtraction` to a row for the Excel file.

    The returned list has one entry per column in :data:`EXCEL_COLUMNS`,
    in the same order, so it can be written straight into the worksheet.

    Args:
        extraction: The combined structured + synthesis result for one article.

    Returns:
        A list of string values aligned with :data:`EXCEL_COLUMNS`.
    """
    combined: dict[str, str] = {
        **extraction.structured.model_dump(),
        **extraction.synthesis.model_dump(),
    }
    return [combined[COLUMN_TO_FIELD[col]] for col in EXCEL_COLUMNS]
