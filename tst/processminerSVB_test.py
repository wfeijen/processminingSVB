from datetime import datetime, timedelta

import polars as pl
import pytest

from processminer import processminerSVB

_ = processminerSVB  # Avoids the Pylance 'not accessed' warning

CASE_SEQUENCES = [
    "AABAABAACAAD",
    "AABAABAACAADAABAABAACAAD",
    "ABCDEFGHIJKL",
    "ABAABAACAADABCDEFGHIJKLAABAABAACAAD",
    "A",
]


def _make_process_log(
    case_column="case_id",
    event_column="activity",
    start_column="start_time",
    end_column="end_time",
    case_ids=None,
):
    if case_ids is None:
        case_ids = [f"C{index}" for index in range(1, len(CASE_SEQUENCES) + 1)]

    cases = []
    activities = []
    starts = []
    ends = []
    amounts = []
    for case_index, (case_id, sequence) in enumerate(zip(case_ids, CASE_SEQUENCES)):
        case_start = datetime(2026, 1, 1) + timedelta(days=case_index)
        for position, activity in enumerate(sequence):
            event_start = case_start + timedelta(minutes=10 * position)
            cases.append(case_id)
            activities.append(activity)
            starts.append(event_start)
            ends.append(event_start + timedelta(minutes=5))
            amounts.append(1)

    return pl.DataFrame(
        {
            case_column: cases,
            event_column: activities,
            start_column: starts,
            end_column: ends,
            "amount": amounts,
        }
    )


@pytest.fixture
def process_log():
    """Fixture with the five representative process cases."""
    return _make_process_log()


@pytest.fixture
def sample_df_custom():
    """Fixture with custom column names for testing .config()."""
    return _make_process_log(
        case_column="case_number",
        event_column="step",
        start_column="begin_time",
        end_column="finish_time",
        case_ids=[1, 2, 3, 4, 5],
    )


def test_case_count_default(process_log):
    """Test that the number of unique cases is counted correctly with defaults."""
    count = process_log.pm.case_count()

    assert count == 5


def test_troughput_time_default(process_log):
    """Test that throughput time is calculated correctly with defaults."""
    result = process_log.pm.troughput_time()

    # Check that the expected columns are present
    assert "case_id" in result.columns
    assert "case_start" in result.columns
    assert "case_end" in result.columns
    assert "throughput_time" in result.columns

    assert result.height == 5


def test_config_and_custom_columns(sample_df_custom):
    """Test that .config() works and supports method chaining."""
    # Test the case count with a custom column name
    count = sample_df_custom.pm.config(case_id="case_number").case_count()
    assert count == 5

    # Test throughput time with fully configured custom columns
    result = (
        sample_df_custom.pm.config(
            case_id="case_number", start_time="begin_time", end_time="finish_time"
        )
        .filter(pl.col("case_number") == 1)  # Also check that Polars chaining works
        .troughput_time()
    )

    assert "case_number" in result.columns
    assert result.height == 1  # Only case 1 remains after filtering

def test_dataframe_namespace_creation():
    df = _make_process_log(
        case_column="case_number",
        event_column="step",
        start_column="begin_time",
        end_column="finish_time",
        case_ids=[1, 2, 3, 4, 5],
    ).pm.config(case_id="case_number", start_time="begin_time", end_time="finish_time")

    assert isinstance(df, processminerSVB)
    assert df.case_count() == 5


def test_filter_precedence_and_inverse(process_log):
    matching = process_log.pm.filter_precedence(["A"], ["B"], max_gap=0)
    inverse = process_log.pm.filter_precedence(["A"], ["B"], inverse=True)

    assert sorted(matching["case_id"].unique().to_list()) == ["C1", "C2", "C3", "C4"]
    assert inverse["case_id"].unique().to_list() == ["C5"]


def test_trim_and_case_and_event_slices(process_log):
    trimmed = process_log.pm.trim(["B"], ["A"])
    head_cases = process_log.pm.head_cases(n=1)
    head_events = process_log.pm.head_event_types(n=1)

    trimmed_c1 = trimmed.filter(pl.col("case_id") == "C1")
    assert trimmed_c1["activity"].to_list() == ["B", "A", "A", "B", "A", "A", "C", "A", "A"]
    assert sorted(trimmed["case_id"].unique().to_list()) == ["C1", "C2", "C4"]
    assert head_cases["case_id"].unique().to_list() == ["C4"]
    assert head_events["activity"].unique().to_list() == ["A"]


def test_aggregations(process_log):
    per_case = process_log.pm.agg_per_case([("amount", "amount_total", "sum")])
    per_event = process_log.pm.agg_per_event_type([("amount", "amount_mean", "mean")])

    assert per_case.sort("case_id")["amount_total"].to_list() == [12, 24, 12, 35, 1]
    assert per_event.filter(pl.col("activity") == "A")["amount_mean"].item() == 1


def test_time_metrics_and_selected_points(process_log):
    full = process_log.pm
    selected = process_log.pm.throughput_time(["B"], ["A"])

    c1_full = full.processing_time().filter(pl.col("case_id") == "C1")
    c1_idle = full.idle_time().filter(pl.col("case_id") == "C1")
    c1_selected = selected.filter(pl.col("case_id") == "C1")

    assert c1_full["processing_time"].item().total_seconds() == 60 * 60
    assert c1_idle["idle_time"].item().total_seconds() == 55 * 60
    assert c1_selected["throughput_time"].item().total_seconds() == 85 * 60


def test_split_and_unravel(process_log):
    parts = process_log.pm.split(["B"], events_preceding=1, events_following=1)
    unraveled = process_log.pm.unravel(["A"])
    unraveled_all = process_log.pm.unravel()

    assert len(parts) == 12
    assert parts[0]["activity"].to_list() == ["A", "B", "A"]
    assert (
        unraveled.filter(pl.col("case_id") == "C1")["activity"].to_list()
        == ["A", "A_1", "B", "A_2", "A_3", "B", "A_4", "A_5", "C", "A_6", "A_7", "D"]
    )
    assert (
        unraveled_all.filter(pl.col("case_id") == "C1")["activity"].to_list()
        == ["A", "A_1", "B", "A_2", "A_3", "B_1", "A_4", "A_5", "C", "A_6", "A_7", "D"]
    )
    assert (
        unraveled_all.filter(pl.col("case_id") == "C2")["activity"].to_list()
        == [
            "A", "A_1", "B", "A_2", "A_3", "B_1", "A_4", "A_5", "C", "A_6", "A_7", "D",
            "A_8", "A_9", "B_2", "A_10", "A_11", "B_3", "A_12", "A_13", "C_1", "A_14", "A_15",
            "D_1",
        ]
    )
