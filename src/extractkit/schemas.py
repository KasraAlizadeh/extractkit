"""Extraction schemas for systematic literature reviews.

The Excel columns are modelled as several Pydantic classes:

* :class:`BibliographicFields`, :class:`StudyContextFields`,
  :class:`ParticipantsFields`, :class:`MeasurementsFields`,
  :class:`StrategiesFields`, and :class:`ToolingFields` are extracted in
  multiple focused passes, so the LLM can give each topic its full
  attention rather than juggling 28 fields at once.
* :class:`SynthesisFields` covers the free-form summary fields produced
  in a separate pass.
* :class:`StructuredFields` is the merged result of all batched passes,
  kept as the single object the orchestrator and tests consume.

Field descriptions are sent to the LLM as part of the JSON schema, so
they double as field-specific extraction instructions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Placeholder used when a value is not present in the article.
NOT_AVAILABLE: str = "N/A"


class BibliographicFields(BaseModel):
    """Pass 1a — basic identifiers."""

    model_config = ConfigDict(extra="forbid")

    article_name: str = Field(
        default=NOT_AVAILABLE,
        description=("Full title of the article exactly as printed. If not stated, return 'N/A'."),
    )
    key_words: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Author-supplied keywords from after the abstract, comma-"
            "separated. If not stated, return 'N/A'."
        ),
    )
    year: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Year of publication as a 4-digit string (e.g. '2023'). If not stated, return 'N/A'."
        ),
    )
    journal: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Full official journal or conference name (e.g. 'Building and "
            "Environment'). If not stated, return 'N/A'."
        ),
    )
    doi: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Digital Object Identifier (e.g. '10.1016/j.buildenv.2023.110123'). "
            "If not stated, return 'N/A'."
        ),
    )
    researchers: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "All authors comma-separated, in order (e.g. 'Smith J, Lee K'). "
            "If not stated, return 'N/A'."
        ),
    )


class StudyContextFields(BaseModel):
    """Pass 1b — geographic and study-context information."""

    model_config = ConfigDict(extra="forbid")

    country: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Country where the research was conducted. If multiple, comma-"
            "separated. If not stated, return 'N/A'."
        ),
    )
    city: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "City or cities where the research was conducted "
            "(e.g. 'Phoenix'). If not stated, return 'N/A'."
        ),
    )
    climate: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Climate classification. Prefer Köppen-Geiger code "
            "(e.g. 'BWh', 'Csa'). Otherwise use the descriptive term "
            "(e.g. 'hot-arid'). If not stated, return 'N/A'."
        ),
    )
    subjects_of_study: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Comfort domain(s) studied. Choose ALL that apply from this "
            "exact list: 'Visual Comfort', 'Acoustic Comfort', 'Outdoor Air "
            "Quality', 'Thermal Comfort'. Comma-separated. If none apply, "
            "return 'N/A'."
        ),
    )
    seasons_of_study: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Season(s) of measurements / surveys (e.g. 'summer', 'summer "
            "and winter', 'July-September'). If not stated, return 'N/A'."
        ),
    )
    spaces_studied: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Type of urban space studied (e.g. 'outdoor plaza', 'urban "
            "canyon', 'park', 'street'). If not stated, return 'N/A'."
        ),
    )


class ParticipantsFields(BaseModel):
    """Pass 1c — survey-taker information."""

    model_config = ConfigDict(extra="forbid")

    age: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Age range, mean, or distribution of survey takers "
            "(e.g. '25-55 years', 'mean 32.4'). If not stated, return 'N/A'."
        ),
    )
    gender: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Gender distribution (e.g. '60% male, 40% female'). If not stated, return 'N/A'."
        ),
    )
    ethnicity: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Ethnic / national background of survey takers if reported. If "
            "not stated, return 'N/A'."
        ),
    )
    behaviours_activity: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Behaviours and activity level. Prefer met values "
            "(e.g. '1.2 met'). Otherwise describe qualitatively "
            "(e.g. 'seated office work', 'walking'). If not stated, "
            "return 'N/A'."
        ),
    )
    clothing_level: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Clothing insulation level. Prefer clo values "
            "(e.g. '0.5 clo'). Otherwise describe qualitatively "
            "(e.g. 'summer attire'). If not stated, return 'N/A'."
        ),
    )


class MeasurementsFields(BaseModel):
    """Pass 1d — measurements, questionnaire details, KPIs."""

    model_config = ConfigDict(extra="forbid")

    numerical_variables: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Quantitative variables measured / calculated (Ta, RH, Tmrt, Tg, "
            "Va, SR, PMV, PPD, UTCI, SET*, PET). Comma-separated. If none, "
            "return 'N/A'."
        ),
    )
    qualitative_variables: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Qualitative variables collected from survey takers (TSV, TCV, "
            "thermal preference, thermal acceptability, visual / acoustic "
            "ratings). Comma-separated. If none, return 'N/A'."
        ),
    )
    questionnaire_extent: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Number of people in the questionnaire / survey "
            "(e.g. '120 participants', 'N=200'). If not stated, return 'N/A'."
        ),
    )
    questionning_time: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "When and how often the questionnaire was administered "
            "(e.g. 'every 15 minutes', 'morning and afternoon'). If not "
            "stated, return 'N/A'."
        ),
    )
    questionnaire_questions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Brief summary of questionnaire topics / scales "
            "(e.g. 'ASHRAE 7-point thermal sensation'). If not stated, "
            "return 'N/A'."
        ),
    )
    kpi: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "KEY PERFORMANCE INDICATORS — calculated comfort indices "
            "(PMV, PPD, UTCI, SET*, PET, WBGT). Comma-separated. If none, "
            "return 'N/A'."
        ),
    )


class StrategiesFields(BaseModel):
    """Pass 1e — environmental and personal control strategies."""

    model_config = ConfigDict(extra="forbid")

    urban_cooling_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Urban-scale cooling strategies investigated (e.g. 'street "
            "trees', 'green roofs', 'cool pavements'). If none, return 'N/A'."
        ),
    )
    personal_cooling_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Personal-scale cooling strategies investigated (e.g. 'handheld "
            "fans', 'umbrellas'). If none, return 'N/A'."
        ),
    )
    urban_heating_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Urban-scale heating strategies investigated (e.g. 'wind "
            "shelters', 'thermal mass'). If none, return 'N/A'."
        ),
    )
    personal_heating_strategies: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Personal-scale heating strategies investigated (e.g. 'heated "
            "clothing', 'hot drinks'). If none, return 'N/A'."
        ),
    )


class ToolingFields(BaseModel):
    """Pass 1f — modelling and simulation tools."""

    model_config = ConfigDict(extra="forbid")

    software_used: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Software, simulation tools, or statistical packages used "
            "(ENVI-met, RayMan, SOLWEIG, ANSYS Fluent, OpenFOAM, SPSS, R, "
            "MATLAB, Python). State if a native/custom tool was developed. "
            "Comma-separated. If none, return 'N/A'."
        ),
    )


class StructuredFields(BaseModel):
    """Aggregated container of every structured field for one article.

    Produced by combining all batched passes (Bibliographic, StudyContext,
    Participants, Measurements, Strategies, Tooling) into one object.
    """

    model_config = ConfigDict(extra="forbid")

    article_name: str = NOT_AVAILABLE
    key_words: str = NOT_AVAILABLE
    year: str = NOT_AVAILABLE
    journal: str = NOT_AVAILABLE
    doi: str = NOT_AVAILABLE
    researchers: str = NOT_AVAILABLE
    country: str = NOT_AVAILABLE
    city: str = NOT_AVAILABLE
    climate: str = NOT_AVAILABLE
    subjects_of_study: str = NOT_AVAILABLE
    seasons_of_study: str = NOT_AVAILABLE
    spaces_studied: str = NOT_AVAILABLE
    age: str = NOT_AVAILABLE
    gender: str = NOT_AVAILABLE
    ethnicity: str = NOT_AVAILABLE
    behaviours_activity: str = NOT_AVAILABLE
    clothing_level: str = NOT_AVAILABLE
    numerical_variables: str = NOT_AVAILABLE
    qualitative_variables: str = NOT_AVAILABLE
    questionnaire_extent: str = NOT_AVAILABLE
    questionning_time: str = NOT_AVAILABLE
    questionnaire_questions: str = NOT_AVAILABLE
    kpi: str = NOT_AVAILABLE
    urban_cooling_strategies: str = NOT_AVAILABLE
    personal_cooling_strategies: str = NOT_AVAILABLE
    urban_heating_strategies: str = NOT_AVAILABLE
    personal_heating_strategies: str = NOT_AVAILABLE
    software_used: str = NOT_AVAILABLE


class SynthesisFields(BaseModel):
    """Summary fields the LLM writes by reading the whole article."""

    model_config = ConfigDict(extra="forbid")

    research_questions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Research questions the article sets out to answer, as one "
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
            "One paragraph (3-6 sentences) summarising study design, "
            "participants, instruments, and analysis approach."
        ),
    )
    notes: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Notable details, limitations, caveats, or future-work flags, "
            "as one paragraph. If none, return 'N/A'."
        ),
    )
    brief_double_click_to_see_all: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "One-paragraph executive summary covering scope, approach, and headline findings."
        ),
    )
    g_m_r_brief: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Goal-Method-Result brief: one flowing paragraph stating "
            "(a) goals, (b) methodology, (c) headline results."
        ),
    )


class ArticleExtraction(BaseModel):
    """Combined result of all extraction passes for one article."""

    structured: StructuredFields
    synthesis: SynthesisFields


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
    """Flatten an :class:`ArticleExtraction` to a row for the Excel file."""
    combined: dict[str, str] = {
        **extraction.structured.model_dump(),
        **extraction.synthesis.model_dump(),
    }
    return [combined[COLUMN_TO_FIELD[col]] for col in EXCEL_COLUMNS]
