"""Extraction schemas for outdoor environmental comfort literature reviews.

The Excel columns are modelled as several Pydantic classes, one per
extraction batch. Splitting the schema this way lets us make focused
API calls per topic area — one for identifiers, one for participants,
one for measurements, and so on — which produces markedly more complete
extractions than a single 39-field call.

Field descriptions are sent to the LLM as part of the JSON schema, so
they double as field-specific extraction instructions. The system prompt
in :mod:`extractkit.llm_client` supplies the wider extraction contract
(NA / UNCERTAIN / MULTIPLE labels, pipe separators, evidence priority,
study-type sensitivity).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Placeholder used when a value is not present in the article. The prompt
# also allows the LLM to return 'UNCERTAIN' (ambiguous / contradictory) or
# 'MULTIPLE' (many values, no clear primary) — both are valid values for
# any field.
NOT_AVAILABLE: str = "NA"


class ClassificationFields(BaseModel):
    """Group A part 1 — article classification and main focus."""

    model_config = ConfigDict(extra="forbid")

    article_classification: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Single best classification of this article: 'Field Study', "
            "'Simulation Study', 'Review', 'Mixed (Field + Simulation)', "
            "'Method / Model Paper', 'Machine Learning / Predictive Study', "
            "'Case Study', or 'Other (specify)'. If multi-type, list both "
            "separated by ' | '. Return 'NA', 'UNCERTAIN', or 'MULTIPLE' if "
            "the article does not clearly fit."
        ),
    )
    main_focus: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Main comfort focus(es) of the article. Choose ALL that apply "
            "from: 'Thermal Comfort', 'Outdoor Air Quality', 'Visual "
            "Comfort', 'Acoustic Comfort', 'Multi-Sensory Comfort', "
            "'Calibration / Assessment Tools', 'Other (specify)'. Separate "
            "multiple values with ' | '. Return 'NA' or 'UNCERTAIN' if not "
            "clear."
        ),
    )


class BibliographicFields(BaseModel):
    """Group A part 2 — bibliographic identifiers."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Full title of the article exactly as published. No truncation. "
            "Return 'NA' or 'UNCERTAIN' if not resolvable."
        ),
    )
    authors: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "All author names as listed (Last, First | Last, First). If "
            "more than 6 authors, list first 6 then 'et al.'. Return 'NA' "
            "or 'UNCERTAIN' if not resolvable."
        ),
    )
    year: str = Field(
        default=NOT_AVAILABLE,
        description="Four-digit publication year. Return 'NA' if not stated.",
    )
    journal: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Full journal name as published, or conference name if "
            "proceedings. Return 'NA' if not stated."
        ),
    )
    doi: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Full DOI with 'https://doi.org/' prefix, or ISBN, or other "
            "unique identifier. Return 'NA' if not stated."
        ),
    )


class StudyContextFields(BaseModel):
    """Group A part 3 — geographic and study-context information."""

    model_config = ConfigDict(extra="forbid")

    country: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Country or countries where the study was conducted. Multi-"
            "country studies use ' | ' separator. If a simulation has no "
            "physical location, return 'NA'."
        ),
    )
    city: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "City or cities where the study was conducted. Multi-city "
            "studies use ' | ' separator. If no physical location, return "
            "'NA'."
        ),
    )
    climate: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Exact climate description stated in the article (Köppen-"
            "Geiger code, descriptive term, or both). If multiple climates, "
            "list all with ' | '. Return 'NA' if not stated."
        ),
    )
    seasons: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Exact season(s) or months studied (e.g., 'Summer | Winter', "
            "'July-August'). If all seasons covered, 'All seasons'. For "
            "reviews spanning multiple periods, write the range. Return "
            "'NA' if not stated."
        ),
    )
    urban_space_types: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Exact types of outdoor spaces studied (e.g., 'street canyon "
            "| park | university courtyard | public square'). Use the "
            "article's own terminology. If simulation-only, describe the "
            "simulated space and mark '(simulated)'. Return 'NA' if "
            "unspecified."
        ),
    )


class ParticipantsFields(BaseModel):
    """Group B — human-subject information (columns 13-18)."""

    model_config = ConfigDict(extra="forbid")

    age: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Age ranges, mean ages, or age group labels as stated "
            "(e.g. '18-30 years (mean 24)', 'Young adults', 'Children "
            "(6-12)'). If assumed for simulation, note '(assumed)'. Return "
            "'NA' if no human subjects."
        ),
    )
    gender: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Gender distribution as stated (e.g., '52% male, 48% female'). "
            "If mixed without data, 'Mixed (unspecified ratio)'. If "
            "assumed for simulation, note '(assumed)'. Return 'NA' if no "
            "human subjects."
        ),
    )
    ethnicity: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Ethnicity, nationality, or participant category (e.g., "
            "'Chinese university students', 'Local residents', 'Tourists "
            "and residents'). Return 'NA' if no human subjects or not "
            "reported."
        ),
    )
    clothing_level: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Both descriptive clothing AND clo values when available. "
            "Format: '[description] ([clo value] clo)'. If assumed for "
            "simulation, note '(assumed)'. Return 'NA' if no human "
            "subjects."
        ),
    )
    activity_during_exposure: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "What participants were doing at measurement / survey "
            "(e.g., 'Walking at 1.2 m/s', 'Sitting on benches'). Multiple "
            "activities separated by ' | '. If assumed for simulation "
            "(e.g., 'walking pedestrian, 1.1 met'), note '(assumed)'. "
            "Return 'NA' if no human subjects."
        ),
    )
    activity_before_exposure: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Activity up to 24 hours before exposure, only if explicitly "
            "stated (e.g., 'participants seated for 15 min before survey', "
            "'>=30 min outdoor exposure required'). Captures pre-"
            "conditioning protocol. Return 'NA' if not reported."
        ),
    )


class MeasurementsFields(BaseModel):
    """Group C — environmental measurements (columns 19-20)."""

    model_config = ConfigDict(extra="forbid")

    variables_quantitative: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Numerical / physical variables measured or simulated, with "
            "units. Preserve article abbreviations exactly. Separate with "
            "' | '. Example: 'Air temperature (Ta, °C) | Relative humidity "
            "(RH, %) | Wind speed (m/s) | Tmrt (°C) | PET (°C) | UTCI (°C) "
            "| SVF'. Include physiological / index variables if numerically "
            "reported. Return 'NA' if none."
        ),
    )
    variables_qualitative: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Non-numerical / perceptual variables recorded, including vote "
            "types and scales. Separate with ' | '. Examples: 'Thermal "
            "Sensation Vote (TSV, ASHRAE 7-point scale: -3 to +3)', "
            "'Thermal Preference Vote (TPV, McIntyre scale)', 'Sun/shade "
            "preference', 'Visual glare assessment'. Return 'NA' if none."
        ),
    )


class SubjectiveComfortFields(BaseModel):
    """Group D — subjective comfort survey information (columns 21-23)."""

    model_config = ConfigDict(extra="forbid")

    questionnaire_extent: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Total questionnaires distributed | valid responses | response "
            "rate. Include survey sessions, locations, or rounds if stated. "
            "Example: '950 questionnaires distributed; 872 valid (92% "
            "response rate)'. Return 'NA' if no survey."
        ),
    )
    survey_time: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Exact time windows, hours, or temporal protocols. Include "
            "seasonal context if stated. Example: '09:00-17:00 daily across "
            "3 weeks in July-August'. Return 'NA' if no survey."
        ),
    )
    questionnaire_questions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Types or categories of questions asked. If quoted verbatim in "
            "the article, list them all separated by ' | '. Return 'NA' if "
            "no survey."
        ),
    )


class CalculatedIndexesFields(BaseModel):
    """Group E — calculated comfort indexes / KPIs (column 24)."""

    model_config = ConfigDict(extra="forbid")

    calculated_indexes: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Every explicitly named index, model output, or derived "
            "indicator. For each: name | equation/formula if stated | "
            "thresholds, neutral ranges, calibrated values if reported. "
            "Example: 'PET (Physiological Equivalent Temperature) — "
            "calculated via RayMan; neutral range 18-23°C | UTCI — derived "
            "from ENVI-met outputs; stress categories applied'. Separate "
            "multiple indexes with ' | '. Return 'NA' if none."
        ),
    )


class StrategiesFields(BaseModel):
    """Group F — PECS: urban and personal control strategies (columns 25-28)."""

    model_config = ConfigDict(extra="forbid")

    urban_cooling: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Urban-scale / design-level interventions to reduce heat or "
            "improve summer comfort, only if explicitly studied, evaluated, "
            "or recommended. Examples: 'vegetation and tree planting | "
            "shading structures | high-albedo materials | water features | "
            "canyon geometry optimization | green roofs / walls | permeable "
            "pavements | wind corridor design'. Return 'NA' if not present."
        ),
    )
    personal_cooling: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Individual-level cooling behaviours / adaptations, only if "
            "explicitly reported or observed. Examples: 'shade-seeking | "
            "clothing adjustment | hat/umbrella/sunglasses | cold drink | "
            "personal fans / misting | reducing activity | timing activity | "
            "indoor retreat'. Return 'NA' if not present."
        ),
    )
    urban_heating: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Urban-scale strategies that preserve / enhance warmth in "
            "winter, only if explicitly studied or recommended. Examples: "
            "'solar access optimization | south-facing orientation | low-"
            "rise or open urban form | dark / high-thermal-mass materials | "
            "wind shelter design'. Return 'NA' if not present."
        ),
    )
    personal_heating: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Individual-level heating behaviours / adaptations, only if "
            "explicitly reported or observed. Examples: 'sun-seeking | "
            "adding clothing layers | warm beverage | increasing activity "
            "intensity | portable heaters or blankets | behavioural / "
            "cultural acclimatization'. Return 'NA' if not present."
        ),
    )


class ModelingFields(BaseModel):
    """Group G — modeling / simulation (column 29)."""

    model_config = ConfigDict(extra="forbid")

    modeling_simulation: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Combined summary of computational modeling / simulation, with "
            "labelled sub-elements: SOFTWARE (name and version), PURPOSE, "
            "INPUTS, OUTPUTS, CALIBRATION / VALIDATION, STATISTICAL / ML "
            "METHODS. Format as pipe-separated flowing text with labels. "
            "Example: 'SOFTWARE: ENVI-met 4.0 | RayMan 1.2 — PURPOSE: "
            "microclimate simulation and PET calculation — INPUTS: Ta, RH, "
            "WS, urban geometry, vegetation — OUTPUTS: PET, Tmrt — "
            "VALIDATION: field measurements R²=0.91 — STATS: linear "
            "regression'. Return 'NA' if no modeling."
        ),
    )


class ReviewFields(BaseModel):
    """Group H — review-study fields (columns 30-35)."""

    model_config = ConfigDict(extra="forbid")

    review_scope: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "For reviews only: topical, geographic, and temporal scope. "
            "Example: 'Outdoor thermal comfort studies globally, 1995-2025, "
            "focus on field surveys and simulation'. Return 'NA' if not a "
            "review."
        ),
    )
    number_of_studies_reviewed: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "For reviews only: exact number of papers included, or range "
            "if given. Return 'NA' if not a review."
        ),
    )
    themes_categories: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "For reviews only: main thematic groupings or classification "
            "categories the review uses. Separate with ' | '. Return 'NA' "
            "if not a review."
        ),
    )
    methods_compared: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "For reviews only: specific methodologies, indices, tools, or "
            "approaches compared across reviewed studies. Separate with "
            "' | '. Return 'NA' if not a review."
        ),
    )
    gaps_identified: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "For reviews only: research gaps, limitations, or future "
            "directions explicitly stated. Use the article's own wording. "
            "Return 'NA' if not a review."
        ),
    )
    conclusions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "For reviews only: main conclusions preserving the article's "
            "own emphasis and language. Return 'NA' if not a review."
        ),
    )


class StructuredFields(BaseModel):
    """Aggregated container of every structured field for one article.

    Produced by merging the output of every batch call. Kept as the
    single object the orchestrator and tests consume so downstream code
    stays unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    # Classification
    article_classification: str = NOT_AVAILABLE
    main_focus: str = NOT_AVAILABLE
    # Bibliographic
    title: str = NOT_AVAILABLE
    authors: str = NOT_AVAILABLE
    year: str = NOT_AVAILABLE
    journal: str = NOT_AVAILABLE
    doi: str = NOT_AVAILABLE
    # Study context
    country: str = NOT_AVAILABLE
    city: str = NOT_AVAILABLE
    climate: str = NOT_AVAILABLE
    seasons: str = NOT_AVAILABLE
    urban_space_types: str = NOT_AVAILABLE
    # Participants
    age: str = NOT_AVAILABLE
    gender: str = NOT_AVAILABLE
    ethnicity: str = NOT_AVAILABLE
    clothing_level: str = NOT_AVAILABLE
    activity_during_exposure: str = NOT_AVAILABLE
    activity_before_exposure: str = NOT_AVAILABLE
    # Measurements
    variables_quantitative: str = NOT_AVAILABLE
    variables_qualitative: str = NOT_AVAILABLE
    # Subjective comfort
    questionnaire_extent: str = NOT_AVAILABLE
    survey_time: str = NOT_AVAILABLE
    questionnaire_questions: str = NOT_AVAILABLE
    # Indexes
    calculated_indexes: str = NOT_AVAILABLE
    # Strategies
    urban_cooling: str = NOT_AVAILABLE
    personal_cooling: str = NOT_AVAILABLE
    urban_heating: str = NOT_AVAILABLE
    personal_heating: str = NOT_AVAILABLE
    # Modeling
    modeling_simulation: str = NOT_AVAILABLE
    # Review
    review_scope: str = NOT_AVAILABLE
    number_of_studies_reviewed: str = NOT_AVAILABLE
    themes_categories: str = NOT_AVAILABLE
    methods_compared: str = NOT_AVAILABLE
    gaps_identified: str = NOT_AVAILABLE
    conclusions: str = NOT_AVAILABLE


class SynthesisFields(BaseModel):
    """Group I — GMR fields, produced in a separate synthesis pass."""

    model_config = ConfigDict(extra="forbid")

    research_questions: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Explicit research questions stated in the article, or implied "
            "questions extracted from stated objectives. Multiple questions "
            "separated by ' | '."
        ),
    )
    key_goals: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "Explicit aims, objectives, or purposes in the article's own "
            "language. Multiple goals separated by ' | '."
        ),
    )
    methodology: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "2-5 sentence description of overall methodology: design, "
            "instruments, sample strategy, analysis approach. Synthesize "
            "from methods section, not the abstract."
        ),
    )
    gmr_brief: str = Field(
        default=NOT_AVAILABLE,
        description=(
            "ONE sentence, max 40 words, summarising Goal + Method + "
            "Result. Format: '[Goal] using [method], finding that "
            "[result].'"
        ),
    )


class ArticleExtraction(BaseModel):
    """Combined result of every extraction pass for one article."""

    structured: StructuredFields
    synthesis: SynthesisFields


# Ordered list of the 39 Excel column headers, matching the exact order
# required by the extraction spec.
EXCEL_COLUMNS: tuple[str, ...] = (
    "Article Classification",
    "Main Focus",
    "Title",
    "Authors",
    "Year",
    "Journal",
    "DOI / Unique Identifier",
    "Country",
    "City",
    "Climate",
    "Season(s)",
    "Urban Space Types",
    "Age",
    "Gender",
    "Ethnicity / Nationality / Participant Group",
    "Clothing Level or Description",
    "Activity During Exposure",
    "Activity up to 24 Hours Before Exposure",
    "Environmental Variables Measured: Quantitative",
    "Environmental Variables Measured: Qualitative",
    "Questionnaire Extent",
    "Survey Time",
    "Questionnaire Questions",
    "Calculated Indexes",
    "Urban Cooling Strategies",
    "Personal Cooling Strategies",
    "Urban Heating Strategies",
    "Personal Heating Strategies",
    "Modeling / Simulation",
    "Review Scope",
    "Number of Studies Reviewed",
    "Themes / Categories",
    "Methods Compared",
    "Gaps Identified",
    "Conclusions",
    "Key Research Questions",
    "Key Goals",
    "Methodology",
    "GMR Brief",
)


COLUMN_TO_FIELD: dict[str, str] = {
    "Article Classification": "article_classification",
    "Main Focus": "main_focus",
    "Title": "title",
    "Authors": "authors",
    "Year": "year",
    "Journal": "journal",
    "DOI / Unique Identifier": "doi",
    "Country": "country",
    "City": "city",
    "Climate": "climate",
    "Season(s)": "seasons",
    "Urban Space Types": "urban_space_types",
    "Age": "age",
    "Gender": "gender",
    "Ethnicity / Nationality / Participant Group": "ethnicity",
    "Clothing Level or Description": "clothing_level",
    "Activity During Exposure": "activity_during_exposure",
    "Activity up to 24 Hours Before Exposure": "activity_before_exposure",
    "Environmental Variables Measured: Quantitative": "variables_quantitative",
    "Environmental Variables Measured: Qualitative": "variables_qualitative",
    "Questionnaire Extent": "questionnaire_extent",
    "Survey Time": "survey_time",
    "Questionnaire Questions": "questionnaire_questions",
    "Calculated Indexes": "calculated_indexes",
    "Urban Cooling Strategies": "urban_cooling",
    "Personal Cooling Strategies": "personal_cooling",
    "Urban Heating Strategies": "urban_heating",
    "Personal Heating Strategies": "personal_heating",
    "Modeling / Simulation": "modeling_simulation",
    "Review Scope": "review_scope",
    "Number of Studies Reviewed": "number_of_studies_reviewed",
    "Themes / Categories": "themes_categories",
    "Methods Compared": "methods_compared",
    "Gaps Identified": "gaps_identified",
    "Conclusions": "conclusions",
    "Key Research Questions": "research_questions",
    "Key Goals": "key_goals",
    "Methodology": "methodology",
    "GMR Brief": "gmr_brief",
}


def extraction_to_row(extraction: ArticleExtraction) -> list[str]:
    """Flatten an :class:`ArticleExtraction` into a row for the Excel file."""
    combined: dict[str, str] = {
        **extraction.structured.model_dump(),
        **extraction.synthesis.model_dump(),
    }
    return [combined[COLUMN_TO_FIELD[col]] for col in EXCEL_COLUMNS]
