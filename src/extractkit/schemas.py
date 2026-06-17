"""Extraction schemas for systematic literature reviews.

The Excel columns are modelled here as two Pydantic classes:

* :class:`StructuredFields` covers the factual fields extracted in the
  first LLM pass with strict JSON-schema validation.
* :class:`SynthesisFields` covers the free-form fields produced in the
  second pass, where the LLM is asked to summarise rather than extract.

Splitting the schema in two lets us use the right prompting style for
each kind of field and keeps the JSON returned by the LLM small enough
to validate reliably.

Field descriptions are sent to the LLM as part of the JSON schema, so
they double as field-specific extraction instructions. Each description
states what to look for, where it typically appears in academic articles,
and the expected output format.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Placeholder used when a value is not present in the article. Centralising
# the constant makes the convention easy to change in one place.
NOT_AVAILABLE: str = "N/A"


class StructuredFields(BaseModel):
    """Factual fields extracted directly from the article.

    Every field is optional at the model level: the LLM is instructed to
    return 'N/A' when a value is not stated in the paper, which is
    preferable to hallucinating a plausible-looking answer.
    """

    model_config = ConfigDict(extra="forbid")

    # --- 1. Basic Identifiers ---

    article_name: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "The full title of the article, exactly as printed on the title "
            "page or in the running header. Preserve original capitalisation. "
            "If not stated, return 'N/A'."
        ),
    )
    key_words: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Author-supplied keywords, usually printed after the abstract "
            "under a 'Keywords' heading. Separate multiple keywords with "
            "commas. If not stated, return 'N/A'."
        ),
    )
    year: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Year of publication as a 4-digit string (e.g. '2023'). Found "
            "on the title page, copyright line, or DOI. If not stated, "
            "return 'N/A'."
        ),
    )
    journal: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Full official name of the journal or conference proceedings "
            "(e.g. 'Building and Environment', not 'B&E'). If not stated, "
            "return 'N/A'."
        ),
    )
    doi: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Digital Object Identifier of the article (e.g. "
            "'10.1016/j.buildenv.2023.110123'). Usually printed on the "
            "title page or in the journal header. If not stated, return 'N/A'."
        ),
    )
    researchers: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "All authors comma-separated, in the order they appear on the "
            "title page (e.g. 'Smith J, Lee K, Brown A'). If not stated, "
            "return 'N/A'."
        ),
    )
    country: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Country where the research was conducted. Look in methods, "
            "study area description, or figure captions. If the study spans "
            "multiple countries, list them comma-separated. If not stated, "
            "return 'N/A'."
        ),
    )
    city: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "City or cities where the research was conducted "
            "(e.g. 'Phoenix', 'Singapore'). Found in methods or study area "
            "sections. If not stated, return 'N/A'."
        ),
    )
    climate: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Climate classification of the study area. Prefer Köppen-Geiger "
            "code if stated (e.g. 'BWh', 'Csa', 'Cfb'). Otherwise use the "
            "descriptive term given (e.g. 'hot-arid', 'humid subtropical'). "
            "If not stated, return 'N/A'."
        ),
    )
    subjects_of_study: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "The COMFORT DOMAIN(S) the article studies. Choose ALL that apply "
            "from this exact list: 'Visual Comfort', 'Acoustic Comfort', "
            "'Outdoor Air Quality', 'Thermal Comfort'. If multiple apply, "
            "separate with commas (e.g. 'Thermal Comfort, Outdoor Air "
            "Quality'). If none of these apply, return 'N/A'. Note: this "
            "field describes the research topic, NOT the participants."
        ),
    )
    seasons_of_study: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Season(s) during which measurements or surveys took place "
            "(e.g. 'summer', 'winter', 'summer and winter', "
            "'July-September'). Found in methods or data-collection "
            "sections. If not stated, return 'N/A'."
        ),
    )
    spaces_studied: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Type of urban space(s) studied (e.g. 'outdoor plaza', "
            "'urban canyon', 'park', 'street', 'pedestrian boulevard', "
            "'courtyard'). Indicate the urban context explicitly. If not "
            "stated, return 'N/A'."
        ),
    )

    # --- 2. Subjects' Personal Information ---

    age: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Age range, mean, or distribution of survey takers "
            "(e.g. '25-55 years', 'mean 32.4 years', '18-65, mean 40'). "
            "Found in participants section. If not stated, return 'N/A'."
        ),
    )
    gender: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Gender distribution of survey takers (e.g. '60% male, 40% "
            "female', 'equal split', '72 male / 48 female'). Found in "
            "participants section. If not stated, return 'N/A'."
        ),
    )
    ethnicity: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Ethnic or national background of survey takers (e.g. 'Chinese', "
            "'mixed European descent', 'local residents'). Often missing from "
            "thermal-comfort papers. If not stated, return 'N/A'."
        ),
    )
    behaviours_activity: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Behaviours and activity level of survey takers. Prefer met "
            "values (e.g. '1.2 met'). If only described qualitatively "
            "(e.g. 'seated office work', 'walking', 'light activity'), use "
            "that description. Sometimes written as 'M' in equations. If "
            "not stated, return 'N/A'."
        ),
    )
    clothing_level: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Clothing insulation level of survey takers. Prefer clo values "
            "(e.g. '0.5 clo', 'Icl = 0.8'). If only described qualitatively "
            "(e.g. 'summer attire', 'business casual', 'light clothing'), "
            "record that description. If not stated, return 'N/A'."
        ),
    )

    # --- 3. Comfort measurements ---

    numerical_variables: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Quantitative variables measured or calculated. Include common "
            "abbreviations (Ta = air temperature, RH = relative humidity, "
            "Tmrt = mean radiant temperature, Tg = globe temperature, Va = "
            "air velocity, SR = solar radiation, PMV, PPD, UTCI, SET*, PET). "
            "Separate with commas. If not stated, return 'N/A'."
        ),
    )
    qualitative_variables: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Qualitative variables collected from survey takers, typically "
            "subjective votes (thermal sensation vote / TSV, thermal "
            "comfort vote / TCV, thermal preference, thermal acceptability, "
            "visual comfort rating, acoustic annoyance). Separate with "
            "commas. If not stated, return 'N/A'."
        ),
    )

    # --- 4. Subject comfort Information ---

    questionnaire_extent: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Number of people who participated in the questionnaire / survey "
            "(e.g. '120 participants', '450 respondents', 'N=200'). Found in "
            "methods or participants section. If not stated, return 'N/A'."
        ),
    )
    questionning_time: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "When and how often the questionnaire was administered "
            "(e.g. 'every 15 minutes', 'once per session', 'morning and "
            "afternoon', 'continuous over 4 weeks'). If not stated, return "
            "'N/A'."
        ),
    )
    questionnaire_questions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Brief summary of the topics or scales covered by the "
            "questionnaire (e.g. 'ASHRAE 7-point thermal sensation, comfort, "
            "preference', 'demographics + 5-point comfort scale'). If not "
            "stated, return 'N/A'."
        ),
    )

    # --- 5. Calculated indexes ---

    kpi: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "KEY PERFORMANCE INDICATORS — the indices used to CALCULATE "
            "comfort levels in the study (e.g. 'PMV', 'PPD', 'UTCI', 'SET*', "
            "'PET', 'WBGT', 'OUT_SET*'). These are calculation outputs, not "
            "raw measurements. Separate multiple KPIs with commas. If none "
            "are used, return 'N/A'."
        ),
    )

    # --- 6. Environmental or Personal Controls ---

    urban_cooling_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Urban-scale cooling strategies investigated (e.g. 'street tree "
            "canopy', 'green roofs', 'cool pavements', 'water features', "
            "'high-albedo surfaces', 'urban geometry modification'). If the "
            "study did not address urban cooling, return 'N/A'."
        ),
    )
    personal_cooling_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Personal-scale cooling strategies investigated (e.g. 'handheld "
            "fans', 'umbrellas', 'cooling vests', 'misting devices', 'cold "
            "drinks', 'clothing adjustment'). If the study did not address "
            "personal cooling, return 'N/A'."
        ),
    )
    urban_heating_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Urban-scale heating strategies investigated (e.g. 'wind "
            "shelters', 'enclosed courtyards', 'thermal mass', 'low-albedo "
            "surfaces'). If the study did not address urban heating, return "
            "'N/A'."
        ),
    )
    personal_heating_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Personal-scale heating strategies investigated (e.g. 'heated "
            "clothing', 'hand warmers', 'hot drinks', 'increased clothing "
            "insulation'). If the study did not address personal heating, "
            "return 'N/A'."
        ),
    )

    # --- 7. Modelling and Simulation ---

    software_used: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Software, simulation tools, or statistical packages used "
            "(e.g. 'ENVI-met', 'RayMan', 'SOLWEIG', 'Ladybug', 'ANSYS "
            "Fluent', 'OpenFOAM', 'SPSS', 'R', 'MATLAB', 'Python'). State "
            "if a native/custom tool was developed. Separate multiple tools "
            "with commas. If no software is mentioned, return 'N/A'."
        ),
    )


class SynthesisFields(BaseModel):
    """Summary fields the LLM writes by reading the whole article.

    Unlike :class:`StructuredFields`, these are not present verbatim in
    the source text; the LLM has to synthesise them.
    """

    model_config = ConfigDict(extra="forbid")

    research_questions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "The research questions the article sets out to answer, as one "
            "concise paragraph. If not stated, return 'N/A'."
        ),
    )
    key_goals: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Stated goals or objectives of the study, as one concise "
            "paragraph. If not stated, return 'N/A'."
        ),
    )
    methodology: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "One concise paragraph (3-6 sentences) summarising the study "
            "design, participants, instruments, and analysis approach."
        ),
    )
    notes: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Notable details, limitations, caveats, or future-work flags "
            "from the article, as one paragraph. If none, return 'N/A'."
        ),
    )
    brief_double_click_to_see_all: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "A one-paragraph executive summary covering scope, approach, "
            "and headline findings of the article."
        ),
    )
    g_m_r_brief: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Goal-Method-Result brief: a single flowing paragraph that "
            "states (a) the goals of the study, (b) the methodology used, "
            "and (c) the headline results, in that order."
        ),
    )


class ArticleExtraction(BaseModel):
    """Combined result of both extraction passes for one article."""

    structured: StructuredFields
    synthesis: SynthesisFields


# Ordered list of the Excel column headers exactly as they appear in
# the template. The order is the column order in the output workbook.
# Pairing this with the Pydantic field names below means we can map
# model values to the correct Excel column without relying on field
# ordering inside Python (which would be fragile).
EXCEL_COLUMNS: tuple[str, ...] = (
    "Article Name",
    "Key Words",
    "Year",
    "Journal",
    "DOI",
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
    "DOI": "doi",
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
