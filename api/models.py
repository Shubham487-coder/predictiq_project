from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Block 1 — Annotated Type Aliases (real bounds from features.parquet)
# ---------------------------------------------------------------------------

# --- Raw sensor readings ---
VoltRaw             = Annotated[float, Field(ge=97.0,   le=256.0,  description="Raw voltage reading (V)")]
RotateRaw           = Annotated[float, Field(ge=138.0,  le=696.0,  description="Raw rotation speed (RPM)")]
PressureRaw         = Annotated[float, Field(ge=51.0,   le=186.0,  description="Raw pressure reading (PSI)")]
VibrationRaw        = Annotated[float, Field(ge=14.0,   le=77.0,   description="Raw vibration reading (mm/s)")]

# --- 24h rolling statistics ---
VoltMean24h         = Annotated[float, Field(ge=155.0,  le=221.0,  description="24h mean voltage (V)")]
VoltStd24h          = Annotated[float, Field(ge=6.0,    le=29.0,   description="24h voltage std dev (V)")]
RotateMean24h       = Annotated[float, Field(ge=265.0,  le=503.0,  description="24h mean rotation (RPM)")]
RotateStd24h        = Annotated[float, Field(ge=18.0,   le=106.0,  description="24h rotation std dev (RPM)")]
PressureMean24h     = Annotated[float, Field(ge=90.0,   le=153.0,  description="24h mean pressure (PSI)")]
PressureStd24h      = Annotated[float, Field(ge=4.0,    le=29.0,   description="24h pressure std dev (PSI)")]
VibrationMean24h    = Annotated[float, Field(ge=35.0,   le=62.0,   description="24h mean vibration (mm/s)")]
VibrationStd24h     = Annotated[float, Field(ge=2.0,    le=13.0,   description="24h vibration std dev (mm/s)")]

# --- 48h rolling statistics ---
VoltMean48h         = Annotated[float, Field(ge=160.0,  le=216.0,  description="48h mean voltage (V)")]
VoltStd48h          = Annotated[float, Field(ge=8.0,    le=30.0,   description="48h voltage std dev (V)")]
RotateMean48h       = Annotated[float, Field(ge=276.0,  le=484.0,  description="48h mean rotation (RPM)")]
RotateStd48h        = Annotated[float, Field(ge=28.0,   le=108.0,  description="48h rotation std dev (RPM)")]
PressureMean48h     = Annotated[float, Field(ge=92.0,   le=151.0,  description="48h mean pressure (PSI)")]
PressureStd48h      = Annotated[float, Field(ge=5.0,    le=29.0,   description="48h pressure std dev (PSI)")]
VibrationMean48h    = Annotated[float, Field(ge=36.0,   le=61.0,   description="48h mean vibration (mm/s)")]
VibrationStd48h     = Annotated[float, Field(ge=2.0,    le=12.0,   description="48h vibration std dev (mm/s)")]

# --- 72h rolling statistics ---
VoltMean72h         = Annotated[float, Field(ge=160.0,  le=208.0,  description="72h mean voltage (V)")]
VoltStd72h          = Annotated[float, Field(ge=9.0,    le=30.0,   description="72h voltage std dev (V)")]
RotateMean72h       = Annotated[float, Field(ge=318.0,  le=478.0,  description="72h mean rotation (RPM)")]
RotateStd72h        = Annotated[float, Field(ge=32.0,   le=107.0,  description="72h rotation std dev (RPM)")]
PressureMean72h     = Annotated[float, Field(ge=94.0,   le=137.0,  description="72h mean pressure (PSI)")]
PressureStd72h      = Annotated[float, Field(ge=6.0,    le=29.0,   description="72h pressure std dev (PSI)")]
VibrationMean72h    = Annotated[float, Field(ge=37.0,   le=55.0,   description="72h mean vibration (mm/s)")]
VibrationStd72h     = Annotated[float, Field(ge=3.0,    le=12.0,   description="72h vibration std dev (mm/s)")]

# --- Error counts (float in data, whole numbers only — enforced by validator) ---
ErrorCount24h       = Annotated[float, Field(ge=0.0,    le=2.0,    description="Error count in last 24h (0.0, 1.0 or 2.0)")]
ErrorCount48h       = Annotated[float, Field(ge=0.0,    le=3.0,    description="Error count in last 48h (0.0 to 3.0)")]
ErrorCount48hMax2   = Annotated[float, Field(ge=0.0,    le=2.0,    description="Error5 count in last 48h (0.0, 1.0 or 2.0)")]
ErrorCount72h       = Annotated[float, Field(ge=0.0,    le=3.0,    description="Error count in last 72h (0.0 to 3.0)")]
ErrorCount72hMax2   = Annotated[float, Field(ge=0.0,    le=2.0,    description="Error5 count in last 72h (0.0, 1.0 or 2.0)")]
ErrorCountTotal72h  = Annotated[float, Field(ge=0.0,    le=5.0,    description="Total errors across all types in 72h")]

# --- Maintenance recency ---
DaysSinceMaint1     = Annotated[float, Field(ge=0.0,    le=492.0,  description="Days since comp1 maintenance")]
DaysSinceMaint2     = Annotated[float, Field(ge=0.0,    le=349.0,  description="Days since comp2 maintenance")]
DaysSinceMaint3     = Annotated[float, Field(ge=0.0,    le=371.0,  description="Days since comp3 maintenance")]
DaysSinceMaint4     = Annotated[float, Field(ge=0.0,    le=395.0,  description="Days since comp4 maintenance")]

# --- Machine identity ---
MachineAge          = Annotated[int,   Field(ge=0,      le=20,     description="Machine age in years (integer)")]
OneHotFlag          = Annotated[int,   Field(ge=0,      le=1,      description="One-hot model flag (0 or 1)")]
FailFlag            = Annotated[int,   Field(ge=0,      le=1,      description="Binary failure flag (0 or 1)")]

# --- Engineered features ---
VoltRotateRatio24h  = Annotated[float, Field(ge=0.33,   le=0.64,   description="24h volt/rotate ratio")]
VibrationCV24h      = Annotated[float, Field(ge=0.04,   le=0.26,   description="24h vibration coefficient of variation")]
HoursSinceLastError = Annotated[float, Field(ge=1.0,    le=9999.0, description="Hours since last error (9999=never)")]


# ---------------------------------------------------------------------------
# Block 2 — Generic Response Envelope
# ---------------------------------------------------------------------------

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard wrapper for every API response.

    Every endpoint returns this shape — no exceptions.
    status is always 'success' or 'error'.
    data holds the result on success, null on error.
    error holds the message on error, null on success.
    """
    status: str = "success"
    data: Optional[T] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

    @classmethod
    def ok(cls, data: T) -> "APIResponse[T]":
        return cls(status="success", data=data)

    @classmethod
    def failure(cls, error: str) -> "APIResponse[None]":
        return cls(status="error", data=None, error=error)


# ---------------------------------------------------------------------------
# Block 3 — ClassifierRequest (56 features + machine_id metadata)
# ---------------------------------------------------------------------------

class ClassifierRequest(BaseModel):
    """
    Input for XGBoost failure classifier.

    All 56 features must match what the XGBoost models were trained on.
    Bounds are from actual min/max in features.parquet — not assumptions.
    Sending values outside these bounds means the data did not come
    from the same pipeline that produced training data.
    machine_id and alert_threshold are metadata — not passed to the model.
    strict=True: no silent type coercion. A string where a float is
    expected is rejected, not converted.
    """

    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "machine_id": 47,
                "alert_threshold": 0.5,
                "volt": 171.2, "rotate": 445.3,
                "pressure": 108.1, "vibration": 41.5,
                "volt_mean_24h": 169.8, "volt_std_24h": 12.1,
                "rotate_mean_24h": 443.1, "rotate_std_24h": 45.2,
                "pressure_mean_24h": 107.5, "pressure_std_24h": 8.3,
                "vibration_mean_24h": 40.9, "vibration_std_24h": 4.1,
                "volt_mean_48h": 170.1, "volt_std_48h": 13.5,
                "rotate_mean_48h": 444.2, "rotate_std_48h": 47.1,
                "pressure_mean_48h": 107.8, "pressure_std_48h": 9.2,
                "vibration_mean_48h": 41.1, "vibration_std_48h": 4.5,
                "volt_mean_72h": 170.5, "volt_std_72h": 14.2,
                "rotate_mean_72h": 444.8, "rotate_std_72h": 48.3,
                "pressure_mean_72h": 108.0, "pressure_std_72h": 9.8,
                "vibration_mean_72h": 41.3, "vibration_std_72h": 4.8,
                "error1_count_24h": 0.0, "error2_count_24h": 1.0,
                "error3_count_24h": 0.0, "error4_count_24h": 0.0,
                "error5_count_24h": 0.0,
                "error1_count_48h": 0.0, "error2_count_48h": 1.0,
                "error3_count_48h": 0.0, "error4_count_48h": 0.0,
                "error5_count_48h": 0.0,
                "error1_count_72h": 0.0, "error2_count_72h": 1.0,
                "error3_count_72h": 0.0, "error4_count_72h": 0.0,
                "error5_count_72h": 0.0,
                "days_since_comp1_maint": 23.0,
                "days_since_comp2_maint": 47.0,
                "days_since_comp3_maint": 12.0,
                "days_since_comp4_maint": 35.0,
                "age": 18,
                "model_model1": 0, "model_model2": 0,
                "model_model3": 1, "model_model4": 0,
                "volt_rotate_ratio_24h": 0.383,
                "vibration_cv_24h": 0.100,
                "error_count_total_72h": 1.0,
                "hours_since_last_error": 24.0,
            }
        }
    )

    # --- Machine identity (metadata, not a model feature) ---
    machine_id: int = Field(..., ge=1, le=100, description="Machine ID (1-100), used for logging only")
    alert_threshold: float = Field(default=0.5, ge=0.0, le=1.0,
                                   description="Probability threshold for triggering alert (default 0.7)")

    # --- Raw sensor readings ---
    volt:      VoltRaw
    rotate:    RotateRaw
    pressure:  PressureRaw
    vibration: VibrationRaw

    # --- 24h rolling means ---
    volt_mean_24h:      VoltMean24h
    rotate_mean_24h:    RotateMean24h
    pressure_mean_24h:  PressureMean24h
    vibration_mean_24h: VibrationMean24h

    # --- 24h rolling standard deviations ---
    volt_std_24h:      VoltStd24h
    rotate_std_24h:    RotateStd24h
    pressure_std_24h:  PressureStd24h
    vibration_std_24h: VibrationStd24h

    # --- 48h rolling means ---
    volt_mean_48h:      VoltMean48h
    rotate_mean_48h:    RotateMean48h
    pressure_mean_48h:  PressureMean48h
    vibration_mean_48h: VibrationMean48h

    # --- 48h rolling standard deviations ---
    volt_std_48h:      VoltStd48h
    rotate_std_48h:    RotateStd48h
    pressure_std_48h:  PressureStd48h
    vibration_std_48h: VibrationStd48h

    # --- 72h rolling means ---
    volt_mean_72h:      VoltMean72h
    rotate_mean_72h:    RotateMean72h
    pressure_mean_72h:  PressureMean72h
    vibration_mean_72h: VibrationMean72h

    # --- 72h rolling standard deviations ---
    volt_std_72h:      VoltStd72h
    rotate_std_72h:    RotateStd72h
    pressure_std_72h:  PressureStd72h
    vibration_std_72h: VibrationStd72h

    # --- Error counts 24h (max 2 for all types) ---
    error1_count_24h: ErrorCount24h
    error2_count_24h: ErrorCount24h
    error3_count_24h: ErrorCount24h
    error4_count_24h: ErrorCount24h
    error5_count_24h: ErrorCount24h

    # --- Error counts 48h (max 3 for types 1-4, max 2 for type 5) ---
    error1_count_48h: ErrorCount48h
    error2_count_48h: ErrorCount48h
    error3_count_48h: ErrorCount48h
    error4_count_48h: ErrorCount48h
    error5_count_48h: ErrorCount48hMax2

    # --- Error counts 72h (max 3 for types 1-4, max 2 for type 5) ---
    error1_count_72h: ErrorCount72h
    error2_count_72h: ErrorCount72h
    error3_count_72h: ErrorCount72h
    error4_count_72h: ErrorCount72h
    error5_count_72h: ErrorCount72hMax2

    # --- Maintenance recency ---
    days_since_comp1_maint: DaysSinceMaint1
    days_since_comp2_maint: DaysSinceMaint2
    days_since_comp3_maint: DaysSinceMaint3
    days_since_comp4_maint: DaysSinceMaint4

    # --- Machine metadata ---
    age: MachineAge

    # --- Model type one-hot flags (exactly one must be 1) ---
    model_model1: OneHotFlag
    model_model2: OneHotFlag
    model_model3: OneHotFlag
    model_model4: OneHotFlag

    # --- Engineered features ---
    volt_rotate_ratio_24h:  VoltRotateRatio24h
    vibration_cv_24h:       VibrationCV24h
    error_count_total_72h:  ErrorCountTotal72h
    hours_since_last_error: HoursSinceLastError

    # ── Field-level validators ──────────────────────────────────────────────

    @field_validator(
        "error1_count_24h", "error2_count_24h", "error3_count_24h",
        "error4_count_24h", "error5_count_24h",
        "error1_count_48h", "error2_count_48h", "error3_count_48h",
        "error4_count_48h", "error5_count_48h",
        "error1_count_72h", "error2_count_72h", "error3_count_72h",
        "error4_count_72h", "error5_count_72h",
        "error_count_total_72h",
    )
    @classmethod
    def must_be_whole_number(cls, v: float) -> float:
        """Error counts are whole numbers stored as float64.
        0.5 errors is not physically possible."""
        if v != int(v):
            raise ValueError(f"Error counts must be whole numbers, got {v}")
        return v

    @field_validator("model_model1", "model_model2", "model_model3", "model_model4")
    @classmethod
    def must_be_binary(cls, v: int) -> int:
        """One-hot flags must be exactly 0 or 1."""
        if v not in (0, 1):
            raise ValueError(f"Model type flag must be 0 or 1, got {v}")
        return v

    # ── Model-level validator ───────────────────────────────────────────────

    @model_validator(mode="after")
    def exactly_one_model_type(self) -> "ClassifierRequest":
        """Exactly one model flag must be 1.

        Zero flags = unknown machine type.
        Two flags = data corruption.
        Both produce undefined XGBoost behavior.
        """
        total = (self.model_model1 + self.model_model2 +
                 self.model_model3 + self.model_model4)
        if abs(total - 1) > 1e-6:
            raise ValueError(
                f"Exactly one of model_model1/2/3/4 must be 1, "
                f"got sum={total}"
            )
        return self

    # ── Helper method ───────────────────────────────────────────────────────

    def to_feature_array(self) -> list[float]:
        """Return features as flat list in exact model training order.
        machine_id and alert_threshold are excluded — metadata only."""
        return [
            self.volt, self.rotate, self.pressure, self.vibration,
            self.volt_mean_24h, self.volt_std_24h,
            self.rotate_mean_24h, self.rotate_std_24h,
            self.pressure_mean_24h, self.pressure_std_24h,
            self.vibration_mean_24h, self.vibration_std_24h,
            self.volt_mean_48h, self.volt_std_48h,
            self.rotate_mean_48h, self.rotate_std_48h,
            self.pressure_mean_48h, self.pressure_std_48h,
            self.vibration_mean_48h, self.vibration_std_48h,
            self.volt_mean_72h, self.volt_std_72h,
            self.rotate_mean_72h, self.rotate_std_72h,
            self.pressure_mean_72h, self.pressure_std_72h,
            self.vibration_mean_72h, self.vibration_std_72h,
            self.error1_count_24h, self.error2_count_24h,
            self.error3_count_24h, self.error4_count_24h,
            self.error5_count_24h,
            self.error1_count_48h, self.error2_count_48h,
            self.error3_count_48h, self.error4_count_48h,
            self.error5_count_48h,
            self.error1_count_72h, self.error2_count_72h,
            self.error3_count_72h, self.error4_count_72h,
            self.error5_count_72h,
            self.days_since_comp1_maint, self.days_since_comp2_maint,
            self.days_since_comp3_maint, self.days_since_comp4_maint,
            self.age,
            self.model_model1, self.model_model2,
            self.model_model3, self.model_model4,
            self.volt_rotate_ratio_24h, self.vibration_cv_24h,
            self.error_count_total_72h, self.hours_since_last_error,
        ]


# ---------------------------------------------------------------------------
# Block 4 — RULRequest (48 timesteps × 9 features)
# ---------------------------------------------------------------------------

class RULRequest(BaseModel):
    """
    Input for CNN-BiLSTM RUL model.

    sensor_sequence: 48 timesteps x 9 features.
    Each row: [volt, rotate, pressure, vibration,
               volt_mean_6h, rotate_mean_6h,
               pressure_mean_6h, vibration_mean_6h,
               error_count_6h]
    strict=True: no silent type coercion.
    """

    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "machine_id": 47,
                "component": "comp2",
                "sensor_sequence": [
                    [171.2, 445.3, 108.1, 41.5,
                     169.8, 443.1, 107.5, 40.9, 0.0]
                ] * 48
            }
        }
    )

    machine_id: int = Field(..., ge=1, le=100)
    component: Literal["comp1", "comp2", "comp3", "comp4"]
    sensor_sequence: List[List[float]] = Field(
        ...,
        description="48 rows x 9 features. Shape must be exactly (48, 9)."
    )

    @field_validator("sensor_sequence")
    @classmethod
    def validate_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 48:
            raise ValueError(
                f"sensor_sequence must have exactly 48 timesteps, got {len(v)}"
            )
        bounds = [
            (97.0,  256.0, "volt"),
            (138.0, 696.0, "rotate"),
            (51.0,  186.0, "pressure"),
            (14.0,   77.0, "vibration"),
            (97.0,  256.0, "volt_mean_6h"),
            (138.0, 696.0, "rotate_mean_6h"),
            (51.0,  186.0, "pressure_mean_6h"),
            (14.0,   77.0, "vibration_mean_6h"),
            (0.0,    5.0,  "error_count_6h"),
        ]
        for i, row in enumerate(v):
            if len(row) != 9:
                raise ValueError(
                    f"Each timestep must have 9 features, "
                    f"got {len(row)} at timestep {i}"
                )
            for j, (lo, hi, name) in enumerate(bounds):
                val = row[j]
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"At timestep {i}, feature '{name}' = {val:.4f} "
                        f"is outside training range [{lo}, {hi}]"
                    )
        return v


class BatchClassifierRequest(BaseModel):
    """Batch classification — up to 100 machines in one call."""
    requests: List[ClassifierRequest] = Field(
        ..., min_length=1, max_length=100
    )


class BatchRULRequest(BaseModel):
    """Batch RUL prediction — up to 100 requests in one call."""
    requests: List[RULRequest] = Field(
        ..., min_length=1, max_length=100
    )


# ---------------------------------------------------------------------------
# Block 5 — Output Schemas (frozen — immutable after construction)
# ---------------------------------------------------------------------------

class ComponentPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    failure_probability: float = Field(..., ge=0.0, le=1.0)
    alert: bool
    status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    threshold_used: float = Field(..., ge=0.0, le=1.0)


class ClassifierResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    machine_id: int
    comp1: ComponentPrediction
    comp2: ComponentPrediction
    comp3: ComponentPrediction
    comp4: ComponentPrediction
    overall_status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    priority: Literal[1, 2, 3, 4]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RULResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    machine_id: int
    component: Literal["comp1", "comp2", "comp3", "comp4"]
    predicted_rul_hours: float = Field(..., ge=0.0, le=100.0)
    day_bucket: Literal[1, 2, 3, 4, 5]
    day_label: str
    interpretation: Literal["CRITICAL", "URGENT", "MONITOR", "HEALTHY"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MachineStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    machine_id: int
    overall_status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    priority: Literal[1, 2, 3, 4]
    components: dict[str, ComponentPrediction]
    rul_estimates: dict[str, RULResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FleetPriorityList(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_machines: int = Field(..., ge=1)
    critical_count: int = Field(..., ge=0)
    urgent_count:   int = Field(..., ge=0)
    monitor_count:  int = Field(..., ge=0)
    healthy_count:  int = Field(..., ge=0)
    priority_list: List[MachineStatus]
    scan_duration_ms: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def counts_must_sum_to_total(self) -> "FleetPriorityList":
        total = (self.critical_count + self.urgent_count +
                 self.monitor_count + self.healthy_count)
        if total != self.total_machines:
            raise ValueError(
                f"Counts must sum to total_machines={self.total_machines}, "
                f"got {total}"
            )
        return self


class HealthStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["healthy", "degraded", "starting"]
    models_loaded: bool
    classifier_models: List[str]
    rul_model: str
    uptime_seconds: float = Field(..., ge=0.0)
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)