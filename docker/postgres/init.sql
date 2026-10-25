-- ============================================================
-- PredictIQ — PostgreSQL Initialisation
-- ============================================================
-- Runs once when the postgres container is first created.
-- Creates the application database and all tables.
-- ============================================================

-- Create the application database (separate from airflow_db)
SELECT 'CREATE DATABASE predictiq_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'predictiq_db')\gexec

-- Create the airflow database
SELECT 'CREATE DATABASE airflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_db')\gexec

-- Connect to predictiq_db and create tables
\connect predictiq_db

-- ── classifier_results ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS classifier_results (
    id                BIGSERIAL PRIMARY KEY,
    machine_id        INTEGER       NOT NULL,
    overall_status    VARCHAR(10)   NOT NULL CHECK (overall_status IN ('HEALTHY','WARNING','CRITICAL')),
    priority          SMALLINT      NOT NULL CHECK (priority BETWEEN 1 AND 4),

    comp1_prob        REAL          NOT NULL,
    comp1_alert       BOOLEAN       NOT NULL,
    comp1_status      VARCHAR(10)   NOT NULL,

    comp2_prob        REAL          NOT NULL,
    comp2_alert       BOOLEAN       NOT NULL,
    comp2_status      VARCHAR(10)   NOT NULL,

    comp3_prob        REAL          NOT NULL,
    comp3_alert       BOOLEAN       NOT NULL,
    comp3_status      VARCHAR(10)   NOT NULL,

    comp4_prob        REAL          NOT NULL,
    comp4_alert       BOOLEAN       NOT NULL,
    comp4_status      VARCHAR(10)   NOT NULL,

    pipeline_run_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cr_machine_id       ON classifier_results (machine_id);
CREATE INDEX IF NOT EXISTS idx_cr_pipeline_run_at  ON classifier_results (pipeline_run_at DESC);
CREATE INDEX IF NOT EXISTS idx_cr_overall_status   ON classifier_results (overall_status);
CREATE INDEX IF NOT EXISTS idx_cr_priority         ON classifier_results (priority);

-- ── rul_results ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rul_results (
    id                    BIGSERIAL PRIMARY KEY,
    machine_id            INTEGER       NOT NULL,
    component             VARCHAR(10)   NOT NULL CHECK (component IN ('comp1','comp2','comp3','comp4')),

    predicted_rul_hours   REAL          NOT NULL,
    day_bucket            SMALLINT      NOT NULL CHECK (day_bucket BETWEEN 1 AND 5),
    day_label             VARCHAR(30)   NOT NULL,
    interpretation        VARCHAR(10)   NOT NULL CHECK (interpretation IN ('CRITICAL','URGENT','MONITOR','HEALTHY')),

    pipeline_run_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rr_machine_id       ON rul_results (machine_id);
CREATE INDEX IF NOT EXISTS idx_rr_pipeline_run_at  ON rul_results (pipeline_run_at DESC);
CREATE INDEX IF NOT EXISTS idx_rr_component        ON rul_results (component);
CREATE INDEX IF NOT EXISTS idx_rr_interpretation   ON rul_results (interpretation);

-- ── pipeline_runs audit table ────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    status          VARCHAR(20)   NOT NULL DEFAULT 'completed',
    machines_count  INTEGER,
    notes           TEXT
);

COMMENT ON TABLE classifier_results IS 'XGBoost failure classification results per machine per pipeline run';
COMMENT ON TABLE rul_results         IS 'CNN-BiLSTM RUL predictions per machine×component per pipeline run';
COMMENT ON TABLE pipeline_runs       IS 'Audit log of completed pipeline executions';