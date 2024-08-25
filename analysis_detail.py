from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union, Optional
from enum import Enum
import datetime
import uuid

class ScoreTypes(str, Enum):
    UNDEFINED = "Undefined"
    DICHOTOMOUS = "Dichotomous"
    CATEGORICAL = "Categorical"
    ORDINAL = "Ordinal"
    CONTINUOUS = "Continuous"
    BOOLEAN = "Boolean"
    CONTINUOUS_GT = "ContinuousGT"
    CONTINUOUS_LT = "ContinuousLT"
    COUNT = "Count"

class Result(BaseModel):
    labels: List[str]
    values: List[float]
    type: ScoreTypes
    units: Optional[str] = None
    imageIndex: Optional[int] = None
    threshold: Optional[float] = None
    group: Optional[str] = None
    id: str

class Model(BaseModel):
    name: str
    version: int

class Pipeline(BaseModel):
    name: str
    version: int

class Point(BaseModel):
    x: int
    y: int

class VisualisationTypes(str, Enum):
    HEATMAP = "heatmap"
    IMAGE = "image"
    BOUNDING_BOX = "boundingBox"
    LABELLED_BOUNDING_BOX = "labelledBoundingBox"
    LABELLED_FIELD = "labelledField"
    LINE = "line"
    TEXT_BOX = "textBox"
    SALIENCY_MAP = "saliencyMap"
    OVERLAY = "overlay"
    ZIPPED_BINARY = "zippedBinaryCube"
    POLYGON = "polygon"

class Position(BaseModel):
    xMax: int
    xMin: int
    yMax: int
    yMin: int

class SaliencyMap(BaseModel):
    coords: Position
    src: str
    type: VisualisationTypes

class TextBox(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    position: Point
    type: VisualisationTypes

class Line(BaseModel):
    colour: str
    dashed: bool
    id: Optional[str] = None
    points: List[Point]
    type: VisualisationTypes

class Heatmap(BaseModel):
    coords: Position
    type: VisualisationTypes
    src: str
    score: float

class BoundingBox(BaseModel):
    coords: Position
    score: float
    type: VisualisationTypes

class LabelledBoundingBox(BaseModel):
    label: str
    coords: Position
    type: VisualisationTypes

class Visualisation(BaseModel):
    featureName: str
    imageIndex: Optional[int] = None
    target: str
    defaultThreshold: Optional[float] = None
    filter: Optional[str] = None
    objects: Union[
        List[TextBox],
        List[Line],
        List[Heatmap],
        List[BoundingBox],
        List[LabelledBoundingBox],
        List[SaliencyMap],
    ]

class AIOutcome(BaseModel):
    name: str
    desc: str
    modality: str
    pipeline: Pipeline
    group: str
    results: List[Result]
    aggregateResults: List[str] = []
    models: List[Model]
    visualisations: List[Visualisation] = []
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    groupIndex: int
    outcomeIndex: int
    nProcessedImages: int
    focusRegion: Optional[Position] = None
    colour: Optional[str] = None

class Tags(str, Enum):
    LEFT_EYE = "left-eye"
    RIGHT_EYE = "right-eye"
    MACULA_CENTERED = "macula-centered"
    DISC_CENTERED = "disc-centered"
    DISC_ONLY = "disc-only"
    STEREO = "stereo"
    INDETERMINATE = "indeterminate"
    MACULA_ONLY = "macula-only"

class InputSourceOriginal(BaseModel):
    imageIndex: List[int] = []
    src: List[str] = []
    x: int
    y: int
    z: int

class InputSourceDownsized(BaseModel):
    imageIndex: List[int] = []
    src: List[str] = []
    x: int
    y: int
    z: int

class InputSource(BaseModel):
    aggregateImages: List = []
    thumbnail: str
    originals: List[InputSourceOriginal] = []
    downsized: List[InputSourceDownsized] = []

class Urgencies(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    URGENT = "urgent"

class VerboseOutcome(BaseModel):
    text: dict
    description: dict
    urgency: Urgencies

class Status(str, Enum):
    ERROR = "error"
    QUEUED = "queued"
    DONE = "done"
    PROCESSING = "processing"
    EMPTY = "empty"

class Eye(str, Enum):
    LEFT = "left"
    RIGHT = "right"

class Modality(str, Enum):
    FUNDUS = "fundus"
    OCT = "oct"

class AnalysisCreate(BaseModel):
    status: Status
    msg: str
    eye: Optional[Eye] = None
    modality: Modality
    filename: Optional[str] = None
    ai_outcomes: List[AIOutcome] = []
    tags_string: Optional[str] = None
    validated_input_name: Optional[str] = None

class AnalysisResults(AnalysisCreate):
    id: uuid.UUID
    analysis_id: uuid.UUID
    study_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    ai_outcomes: List[AIOutcome] = []
    created: datetime.datetime
    uploaded: datetime.datetime
    tags: Optional[List[Tags]] = None
    validated_input_name: Optional[str] = None
    input_src: Optional[InputSource] = None
    verbose_outcomes: Optional[List[VerboseOutcome]] = None

    model_config = ConfigDict(from_attributes=True)
