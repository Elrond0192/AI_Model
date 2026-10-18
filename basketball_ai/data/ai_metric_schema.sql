-- Metric Rating Engine storage contract.
-- Safe to apply repeatedly. Does not alter HoopmetricsEngine or AI_Source.
BEGIN;

CREATE SCHEMA IF NOT EXISTS "AI";

CREATE TABLE IF NOT EXISTS "AI"."MetricDefinition" (
    metric text PRIMARY KEY,
    source_column text NOT NULL,
    direction text NOT NULL CHECK (direction IN ('higher_better', 'lower_better')),
    zero_reference double precision NOT NULL DEFAULT 0,
    rating_enabled boolean NOT NULL DEFAULT true,
    replacement_reference double precision,
    description text NOT NULL DEFAULT '',
    value_label text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS "AI"."MetricDistribution" (
    metric text NOT NULL REFERENCES "AI"."MetricDefinition"(metric) ON UPDATE CASCADE,
    league_key text NOT NULL,
    season integer NOT NULL,
    competition text NOT NULL,
    population_type text NOT NULL,
    population_source text NOT NULL,
    fallback_used boolean NOT NULL DEFAULT false,
    sample_size integer NOT NULL CHECK (sample_size >= 0),
    min double precision NOT NULL,
    p10 double precision NOT NULL,
    p25 double precision NOT NULL,
    p40 double precision NOT NULL,
    p50 double precision NOT NULL,
    p60 double precision NOT NULL,
    p75 double precision NOT NULL,
    p90 double precision NOT NULL,
    p97_5 double precision NOT NULL,
    max double precision NOT NULL,
    mean double precision NOT NULL,
    stddev double precision NOT NULL CHECK (stddev >= 0),
    quality text NOT NULL CHECK (quality IN ('high', 'medium', 'low', 'insufficient')),
    distribution_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT metric_distribution_context_pk PRIMARY KEY
        (metric, league_key, season, competition, population_type, distribution_version)
);

CREATE TABLE IF NOT EXISTS "AI"."MetricRating" (
    metric text NOT NULL REFERENCES "AI"."MetricDefinition"(metric) ON UPDATE CASCADE,
    player_global_id text NOT NULL,
    player_name text NOT NULL DEFAULT '',
    league_key text NOT NULL,
    season integer NOT NULL,
    competition text NOT NULL,
    value double precision NOT NULL,
    percentile double precision CHECK (percentile IS NULL OR percentile BETWEEN 0 AND 1),
    zscore double precision,
    tier integer CHECK (tier IS NULL OR tier BETWEEN 1 AND 8),
    label text,
    quality text NOT NULL CHECK (quality IN ('high', 'medium', 'low', 'insufficient')),
    population_source text NOT NULL,
    fallback_used boolean NOT NULL DEFAULT false,
    distribution_version text NOT NULL,
    rating_version text NOT NULL,
    above_reference boolean,
    below_reference boolean,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT metric_rating_context_pk PRIMARY KEY
        (metric, player_global_id, league_key, season, competition, distribution_version, rating_version)
);

CREATE INDEX IF NOT EXISTS metric_distribution_lookup_idx
    ON "AI"."MetricDistribution" (metric, league_key, season, competition, distribution_version);

CREATE INDEX IF NOT EXISTS metric_rating_player_context_idx
    ON "AI"."MetricRating" (player_global_id, league_key, season, competition);

CREATE INDEX IF NOT EXISTS metric_rating_metric_context_idx
    ON "AI"."MetricRating" (metric, league_key, season, competition, distribution_version);

COMMIT;
