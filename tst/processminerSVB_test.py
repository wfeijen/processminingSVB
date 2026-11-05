import polars as pl
import pytest

from src.processminer.processminer import processminerSVB

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


def test_aantal_cases_default(sample_df_default):
    """Test of het aantal unieke cases correct wordt geteld met defaults."""
    aantal = sample_df_default.pm.aantal_cases()

    assert aantal == 2


def test_bereken_doorlooptijd_default(sample_df_default):
    """Test of de doorlooptijd correct wordt berekend met defaults."""
    result = sample_df_default.pm.bereken_doorlooptijd()

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
    aantal = sample_df_custom.pm.config(case_id="dossier").aantal_cases()
    assert aantal == 2

    # Test doorlooptijd met volledig geconfigureerde custom kolommen
    result = (
        sample_df_custom.pm.config(
            case_id="dossier",  start_time="begin", end_time="einde"
        )
        .filter(pl.col("dossier") == 1)  # Ook controleren of Polars chaining werkt
        .bereken_doorlooptijd()
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




