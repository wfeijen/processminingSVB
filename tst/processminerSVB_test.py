import polars as pl
import pytest

from processminer import processminerSVB

_ = processminerSVB  # Voorkomt Pylance melding 'not accessed'
@pytest.fixture
def sample_df_default():
    """Fixture met standaard kolomnamen voor tests."""
    return pl.DataFrame(
        {
            "case_id": ["C1", "C1", "C2", "C2"],
            "activity": ["A", "B", "A", "C"],
            "start_time": pl.Series(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 09:30:00",
                ]
            ).str.to_datetime(),
            "end_time": pl.Series(
                [
                    "2026-01-01 10:30:00",
                    "2026-01-01 11:30:00",
                    "2026-01-01 09:15:00",
                    "2026-01-01 10:00:00",
                ]
            ).str.to_datetime(),
        }
    )


@pytest.fixture
def sample_df_custom():
    """Fixture met afwijkende kolomnamen om .config() te testen."""
    return pl.DataFrame(
        {
            "dossier": [1, 1, 2],
            "stap": ["X", "Y", "X"],
            "begin": pl.Series(
                [
                    "2026-06-01 08:00:00",
                    "2026-06-01 09:00:00",
                    "2026-06-01 10:00:00",
                ]
            ).str.to_datetime(),
            "einde": pl.Series(
                [
                    "2026-06-01 08:30:00",
                    "2026-06-01 09:30:00",
                    "2026-06-01 11:00:00",
                ]
            ).str.to_datetime(),
        }
    )


@pytest.fixture
def process_log():
    return pl.DataFrame(
        {
            "case_id": ["C1", "C1", "C1", "C2", "C2"],
            "activity": ["A", "B", "A", "C", "A"],
            "start_time": pl.Series(
                [
                    "2026-01-01 08:00:00",
                    "2026-01-01 08:20:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:30:00",
                ]
            ).str.to_datetime(),
            "end_time": pl.Series(
                [
                    "2026-01-01 08:10:00",
                    "2026-01-01 08:35:00",
                    "2026-01-01 09:10:00",
                    "2026-01-01 10:15:00",
                    "2026-01-01 10:40:00",
                ]
            ).str.to_datetime(),
            "amount": [1, 2, 3, 4, 5],
        }
    )


def test_case_count_default(sample_df_default):
    """Test of het aantal unieke cases correct wordt geteld met defaults."""
    aantal = sample_df_default.pm.case_count()

    assert aantal == 2


def test_troughput_time_default(sample_df_default):
    """Test of de doorlooptijd correct wordt berekend met defaults."""
    result = sample_df_default.pm.troughput_time()

    # Controleer of de juiste kolommen aanwezig zijn
    assert "case_id" in result.columns
    assert "case_start" in result.columns
    assert "case_end" in result.columns
    assert "doorlooptijd" in result.columns

    # Er moeten 2 rijen zijn (1 voor C1, 1 voor C2)
    assert result.height == 2


def test_config_and_custom_columns(sample_df_custom):
    """Test of .config() werkt en method chaining mogelijk maakt."""
    # Test aantal cases met custom kolomnaam
    aantal = sample_df_custom.pm.config(case_id="dossier").case_count()
    assert aantal == 2

    # Test doorlooptijd met volledig geconfigureerde custom kolommen
    result = (
        sample_df_custom.pm.config(
            case_id="dossier",  start_time="begin", end_time="einde"
        )
        .filter(pl.col("dossier") == 1)  # Ook controleren of Polars chaining werkt
        .troughput_time()
    )

    assert "dossier" in result.columns
    assert result.height == 1  # Alleen dossier 1 na filtering

def test_ontstaan_dataframe():
    df = pl.DataFrame(
        {
            "dossier": [1, 1, 2],
            "stap": ["X", "Y", "X"],
            "begin": pl.Series(
                [
                    "2026-06-01 08:00:00",
                    "2026-06-01 09:00:00",
                    "2026-06-01 10:00:00",
                ]
            ).str.to_datetime(),
            "einde": pl.Series(
                [
                    "2026-06-01 08:30:00",
                    "2026-06-01 09:30:00",
                    "2026-06-01 11:00:00",
                ]
            ).str.to_datetime(),
        },
    ).pm.config(case_id="dossier", start_time="begin", end_time="einde")

    assert isinstance(df, processminerSVB)


def test_filter_precedence_and_inverse(process_log):
    matching = process_log.pm.filter_precedence(["A"], ["B"], max_gap=0)
    inverse = process_log.pm.filter_precedence(["A"], ["B"], inverse=True)

    assert matching["case_id"].unique().to_list() == ["C1"]
    assert inverse["case_id"].unique().to_list() == ["C2"]


def test_trim_and_case_and_event_slices(process_log):
    trimmed = process_log.pm.trim(["B"], ["A"])
    head_cases = process_log.pm.head_cases(n=1)
    head_events = process_log.pm.head_event_types(n=1)

    assert trimmed.select("activity").to_series().to_list() == ["B", "A"]
    assert head_cases["case_id"].unique().to_list() == ["C1"]
    assert head_events["activity"].unique().to_list() == ["A"]


def test_aggregations(process_log):
    per_case = process_log.pm.agg_per_case([("amount", "amount_total", "sum")])
    per_event = process_log.pm.agg_per_event_type([("amount", "amount_mean", "mean")])

    assert per_case.sort("case_id")["amount_total"].to_list() == [6, 9]
    assert per_event.filter(pl.col("activity") == "A")["amount_mean"].item() == 3


def test_time_metrics_and_selected_points(process_log):
    full = process_log.pm
    selected = process_log.pm.throughput_time(["B"], ["A"])

    c1_full = full.processing_time().filter(pl.col("case_id") == "C1")
    c1_idle = full.idle_time().filter(pl.col("case_id") == "C1")
    c1_selected = selected.filter(pl.col("case_id") == "C1")

    assert c1_full["processing_time"].item().total_seconds() == 35 * 60
    assert c1_idle["idle_time"].item().total_seconds() == 35 * 60
    assert c1_selected["throughput_time"].item().total_seconds() == 50 * 60


def test_split_and_unravel(process_log):
    parts = process_log.pm.split(["B"], events_preceding=1, events_following=1)
    unraveled = process_log.pm.unravel(["A"])

    assert len(parts) == 1
    assert parts[0]["activity"].to_list() == ["A", "B", "A"]
    assert unraveled["activity"].to_list() == ["A", "B", "A_2", "C", "A_1"]

