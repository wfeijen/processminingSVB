from __future__ import annotations

from datetime import timedelta
from math import ceil

import polars as pl


@pl.api.register_dataframe_namespace("pm")
class processminerSVB:
    def __init__(self, df: pl.DataFrame):
        self._df = df

        # Standaard kolomnamen
        self._case_col = "case_id"
        self._event_col = "activity"
        self._start_col = "start_time"
        self._end_col = "end_time"

# Internals #################################################################################

    def _case_groups(self) -> list[pl.DataFrame]:
        """Return each case's rows in their original event-log order.

        Returns:
            list[pl.DataFrame]: list with dataframes per case, in the original order of the events.
        """
        indexed = self._df.with_row_index("__pm_row_index")
        return indexed.partition_by(self._case_col, maintain_order=True)

    def _filter_rows(self, row_indices: set[int]) -> pl.DataFrame:
        """Return the original rows whose row indices are selected."""
        return self._df.filter(
            pl.Series(
                "__pm_selected",
                [index in row_indices for index in range(self._df.height)],
                dtype=pl.Boolean,
            )
        )

    def _filter_values(self, column: str, values: list[object]) -> pl.DataFrame:
        selected_values = [value for value in values if value is not None]
        predicate = pl.col(column).is_in(selected_values)
        if any(value is None for value in values):
            predicate = predicate | pl.col(column).is_null()
        return self._df.filter(predicate)

    def _select_cases(self, n: int, percentage: float, total: int) -> int:
        if n < -1:
            raise ValueError("n moet -1 (alle) of een niet-negatief aantal zijn.")
        if not 0 <= percentage <= 1:
            raise ValueError("percentage moet tussen 0 en 1 liggen.")
        percentage_count = ceil(total * percentage)
        return percentage_count if n == -1 else min(n, percentage_count)

    def _ranked_cases(self, descending: bool) -> list[tuple[object, int]]:
        ranked: list[tuple[object, int]] = []
        for case in self._case_groups():
            case_id = case[self._case_col][0]
            ranked.append((case_id, case.height))
        return sorted(ranked, key=lambda item: item[1], reverse=descending)

    def _ranked_event_types(self, descending: bool) -> list[tuple[object, int]]:
        counts: dict[object, int] = {}
        for event in self._df.get_column(self._event_col).to_list():
            counts[event] = counts.get(event, 0) + 1
        ranked = list(counts.items())
        return sorted(ranked, key=lambda item: item[1], reverse=descending)

    def _aggregate(
        self, aggregation_rules: list[tuple[str, str, str]], group_column: str
    ) -> pl.DataFrame:
        aggregations = {
            "count": lambda column: column.count(),
            "sum": lambda column: column.sum(),
            "mean": lambda column: column.mean(),
            "min": lambda column: column.min(),
            "max": lambda column: column.max(),
            "first": lambda column: column.first(),
            "last": lambda column: column.last(),
            "n_unique": lambda column: column.n_unique(),
            "median": lambda column: column.median(),
        }
        expressions = []
        for source_column, output_column, function in aggregation_rules:
            if source_column not in self._df.columns:
                raise ValueError(f"Kolom '{source_column}' bestaat niet in de DataFrame.")
            if function not in aggregations:
                supported = ", ".join(aggregations)
                raise ValueError(f"Onbekende aggregatiefunctie '{function}'. Kies uit: {supported}.")
            expressions.append(aggregations[function](pl.col(source_column)).alias(output_column))
        if not expressions:
            return self._df.select(group_column).unique(maintain_order=True)
        return self._df.group_by(group_column, maintain_order=True).agg(expressions)

    def _time_window(
        self, starting_points: list[str], ending_points: list[str]
    ) -> list[tuple[pl.DataFrame, int, int]]:
        """Return inclusive row-order windows between the selected event boundaries."""
        if (starting_points or ending_points) and self._event_col not in self._df.columns:
            raise ValueError(
                f"Kolom '{self._event_col}' is nodig om start- of eindpunten te selecteren."
            )
        windows = []
        for case in self._case_groups():
            activities = (
                case.get_column(self._event_col).to_list()
                if self._event_col in self._df.columns
                else []
            )
            starts = (
                [index for index, event in enumerate(activities) if event in starting_points]
                if starting_points
                else [0]
            )
            ends = (
                [index for index, event in enumerate(activities) if event in ending_points]
                if ending_points
                else [case.height - 1]
            )
            if starts and ends:
                first, last = starts[0], ends[-1]
                if first <= last:
                    windows.append((case, first, last))
        return windows

    def _time_metrics(
        self, starting_points: list[str], ending_points: list[str], metric: str
    ) -> pl.DataFrame:
        records = []
        for case, first, last in self._time_window(starting_points, ending_points):
            events = case.slice(first, last - first + 1)
            starts = events.get_column(self._start_col).to_list()
            ends = events.get_column(self._end_col).to_list()
            valid_starts = [value for value in starts if value is not None]
            valid_ends = [value for value in ends if value is not None]
            case_start = (
                starts[0]
                if starting_points
                else min(valid_starts, default=None)
            )
            case_end = (
                ends[-1]
                if ending_points
                else max(valid_ends, default=None)
            )

            if metric == "throughput":
                if case_start is None or case_end is None:
                    continue
                duration = case_end - case_start
            elif metric == "processing":
                durations = [
                    end - start
                    for start, end in zip(starts, ends)
                    if start is not None and end is not None
                ]
                duration = sum(durations, start=timedelta()) if durations else None
            else:
                gaps = [
                    next_start - previous_end
                    for previous_end, next_start in zip(ends, starts[1:])
                    if previous_end is not None
                    and next_start is not None
                    and next_start > previous_end
                ]
                duration = sum(gaps, start=timedelta())
            records.append(
                {
                    self._case_col: case[self._case_col][0],
                    "case_start": case_start,
                    "case_end": case_end,
                    "duration": duration,
                }
            )

        duration_column = {
            "throughput": "throughput_time",
            "processing": "processing_time",
            "idle": "idle_time",
        }[metric]
        if records:
            return pl.DataFrame(records).rename({"duration": duration_column})
        return pl.DataFrame(
            schema={
                self._case_col: self._df.schema[self._case_col],
                "case_start": self._df.schema[self._start_col],
                "case_end": self._df.schema[self._end_col],
                duration_column: pl.Duration,
            }
        )

    def __getattr__(self, name: str):
        """Zorgt ervoor dat methoden van de onderliggende DataFrame kunnen worden aangeroepen via de namespace wrapper.

        Args:
            name (str): De naam van de methode die aangeroepen moet worden.

        Returns:
            _type_: Het resultaat van de aangeroepen methode weer voorzien van de namespace wrapper.
        """
        # Haal het attribuut op van de onderliggende DataFrame als er gevraagd wordt naar
        # een attribuut dat niet in de wrapper zit.
        attribute = getattr(self._df, name)
        if not callable(attribute):
            return attribute

        # Als het attribuut een callable is, maak een wrapper functie die de aanroep doorstuurt
        # naar de onderliggende DataFrame.
        def delegated(*args, **kwargs):
            """Doet de daadwerkelijke aanroep naar een Polars methode, atrribuut of wat dan ook

            En pakt het resultaat vervolgens weer netjes in in de pm namespace als het resultaat een dataframe is.

            Returns:
                _type_: dataframe met pm namespace of een ander type als het resultaat geen dataframe is.
            """
            result = attribute(*args, **kwargs)
            if isinstance(result, pl.DataFrame):
                chained = type(self)(result)
                chained._case_col = self._case_col
                chained._event_col = self._event_col
                chained._start_col = self._start_col
                chained._end_col = self._end_col
                return chained
            return result

        return delegated

# Public API #################################################################################

# Initialization and configuration methods ###########################################################
    def config(
        self,
        case_id: str = None,
        event_id: str = None,
        start_time: str = None,
        end_time: str = None,
    ):
        """Checkt of de kolomnaam bestaat en kent de betekenis toe aan de kolomnaam._

        Args:
            case_id (str, optional): _description_. Defaults to None.
            event_id (str, optional): _description_. Defaults to None.
            start_time (str, optional): _description_. Defaults to None.
            end_time (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        colomn_names = self._df.columns

        if case_id:
            if case_id not in colomn_names:
                raise ValueError(f"case_id kolom '{case_id}' bestaat niet in de DataFrame.")
            self._case_col = case_id
        if event_id:
            if event_id not in colomn_names:
                raise ValueError(f"event_id kolom '{event_id}' bestaat niet in de DataFrame.")
            self._event_col = event_id
        if start_time:
            if start_time not in colomn_names:
                raise ValueError(f"start_time kolom '{start_time}' bestaat niet in de DataFrame.")
            self._start_col = start_time
        if end_time:
            if end_time not in colomn_names:
                raise ValueError(f"end_time kolom '{end_time}' bestaat niet in de DataFrame.")
            self._end_col = end_time
        return self  # Geeft self terug zodat je kunt chainen


# Filters ###########################################################################################

    def filter_precedence(
        self, antecedents: list[str], consequents: list[str], inverse: bool = False, max_gap: int = -1
    ) -> pl.DataFrame:
        """Filtert cases met de opgegeven volgordes op basis van de opgegeven antecedenten en consequents.

        Een case voldoet als ten minste een antecedent gevolgd wordt door een consequent.
        max_gap telt alleen events die strikt tussen beide events staan.

        Args:
            antecedents (list[str]): De antecedenten waarop gefilterd moet worden.
            consequents (list[str]): De consequents waarop gefilterd moet worden.
            inverse (bool, optional): Indien True, worden de cases die niet voldoen aan de filtercriteria meegenomen. Defaults to False.
            max_gap (int, optional): Het maximale aantal events dat tussen het antecedent en consequent mag zitten. Defaults to -1 (geen limiet).

        Returns:
            pl.DataFrame: Gefilterde DataFrame met alleen de cases die voldoen aan de filtercriteria.
        """  # noqa: E501
        if not antecedents or not consequents:
            raise ValueError("antecedents en consequents mogen niet leeg zijn.")
        if max_gap < -1:
            raise ValueError("max_gap moet -1 (geen limiet) of een niet-negatief aantal zijn.")

        matching_rows: set[int] = set()
        for case in self._case_groups():
            activities = case.get_column(self._event_col).to_list()
            antecedent_positions = [
                index for index, event in enumerate(activities) if event in antecedents
            ]
            consequent_positions = [
                index for index, event in enumerate(activities) if event in consequents
            ]
            matches = any(
                antecedent < consequent
                and (max_gap == -1 or consequent - antecedent - 1 <= max_gap)
                for antecedent in antecedent_positions
                for consequent in consequent_positions
            )
            if matches != inverse:
                matching_rows.update(case.get_column("__pm_row_index").to_list())
        return self._filter_rows(matching_rows)

    def trim(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Trimt de DataFrame op basis van de opgegeven start- en eindpunten.

        Per case wordt de eerste matchende start en de laatste matchende eindactiviteit gebruikt.
        De grens-events blijven behouden; zonder opgegeven grens wordt de casegrens gebruikt.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met alleen de events tussen de opgegeven start- en eindpunten.
        """
        selected: set[int] = set()
        for case, first, last in self._time_window(starting_points, ending_points):
            selected.update(case.slice(first, last - first + 1).get_column("__pm_row_index").to_list())
        return self._filter_rows(selected)


    def head_cases(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de eerste n of de eerste percentage van de cases terug in de DataFrame.

        Sorteren en vervolgens head of tail helpt om in te zoomen op meest voorkomede of langst lopende cases enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende cases worden geselecteerd.
        head_cases sorteert op het aantal events per case, aflopend.

        Args:
            n (int, optional): Aantal over te houden cases. Defaults to -1 (all).
            percentage (float, optional): Percentage van de cases dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Geslicede DataFrame waar alleen de cases tussen de opgegeven start- en eindindex in zitten.
        """
        ranked = self._ranked_cases(descending=True)
        count = self._select_cases(n, percentage, len(ranked))
        return self._filter_values(self._case_col, [case_id for case_id, _ in ranked[:count]])

    def tail_cases(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de laatste n of de laatste percentage van de cases terug in de DataFrame.

        Sorteren en vervolgens head of tail helpt om in te zoomen op meest voorkomede of langst lopende cases enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende cases worden geselecteerd.
        tail_cases sorteert op het aantal events per case, oplopend.

        Args:
            n (int, optional): Aantal over te houden cases. Defaults to -1 (all).
            percentage (float, optional): Percentage van de cases dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Geslicede DataFrame waar alleen de cases tussen de opgegeven start- en eindindex in zitten.
        """
        ranked = self._ranked_cases(descending=False)
        count = self._select_cases(n, percentage, len(ranked))
        return self._filter_values(self._case_col, [case_id for case_id, _ in ranked[:count]])

    def head_event_types(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de eerste n of de eerste percentage van de eventsoorten terug in de DataFrame.

        De eventsoorten worden aflopend op aantal voorkomens gesorteerd.

        Indien start of einde -1 zijn dan is aan die kant geen limiet en wordt alles meegenomen.
        Sorteren en vervolgens slicen helpt om in te zoomen op meest voorkomede of langst durende soorten events enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende eventsoorten worden geselecteerd.

        Args:
            n (int, optional): Aantal over te houden eventsoorten. Defaults to -1 (all).
            percentage (float, optional): Percentage van de eventsoorten dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Gefilterd dataframe.
        """  # noqa: E501
        ranked = self._ranked_event_types(descending=True)
        count = self._select_cases(n, percentage, len(ranked))
        selected = [event for event, _ in ranked[:count]]
        return self._filter_values(self._event_col, selected)

    def tail_event_types(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de laatste n of de laatste percentage van de eventsoorten terug in de DataFrame.

        De eventsoorten worden oplopend op aantal voorkomens gesorteerd.

        Indien start of einde -1 zijn dan is aan die kant geen limiet en wordt alles meegenomen.
        Sorteren en vervolgens slicen helpt om in te zoomen op meest voorkomede of langst durende soorten events enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende eventsoorten worden geselecteerd.

        Args:
            n (int, optional): Aantal over te houden eventsoorten. Defaults to -1 (all).
            percentage (float, optional): Percentage van de eventsoorten dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Gefilterd dataframe.
        """  # noqa: E501
        ranked = self._ranked_event_types(descending=False)
        count = self._select_cases(n, percentage, len(ranked))
        selected = [event for event, _ in ranked[:count]]
        return self._filter_values(self._event_col, selected)


# Enrichement ############################################################################################

    def troughput_time(self) -> pl.DataFrame:
        """Berekent de totale doorlooptijd per case."""
        return self._time_metrics([], [], "throughput").rename(
            {"throughput_time": "doorlooptijd"}
        )

    def case_count(self) -> int:
        """Geeft het unieke aantal cases terug."""
        return self._df.select(pl.col(self._case_col).n_unique()).item()

    def agg_per_case(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggegreert de DataFrame op basis van aggregatie regels over de case_id kolom.

        Elke regel is (bronkolom, uitvoerkolom, functie), bijvoorbeeld
        ("amount", "total_amount", "sum").

        Returns:
            pl.DataFrame: Geaggregeerde DataFrame.
        """
        return self._aggregate(aggregation_rules, self._case_col)


    def agg_per_event_type(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggegreert de DataFrame op basis van aggregatie regels over de event_id kolom.

        Elke regel is (bronkolom, uitvoerkolom, functie), bijvoorbeeld
        ("amount", "average_amount", "mean").

        Returns:
            pl.DataFrame: Geaggregeerde DataFrame.
        """  # noqa: E501
        return self._aggregate(aggregation_rules, self._event_col)



    def throughput_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Bereken de doorlooptijd per case.

        Indien starting_points en of ending_points zijn opgegeven, wordt de doorlooptijd berekend tussen deze punten.
        Niet opgegeven grenzen vallen terug op de eerste respectievelijk laatste event van de case.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met case_id, case_start, case_end en throughput_time kolommen.
        """
        return self._time_metrics(starting_points, ending_points, "throughput")


    def processing_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Bereken de verwerkingstijd per case.

        De verwerkingstijd is de som van end_time - start_time per event in de gekozen case-window.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met case_id, case_start, case_end en processing_time kolommen.
        """
        return self._time_metrics(starting_points, ending_points, "processing")


    def idle_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Bereken de inactieve tijd per case.

        De inactieve tijd is de som van positieve gaps tussen een event-end en het volgende event-start
        binnen de gekozen case-window.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met case_id, case_start, case_end en idle_time kolommen.
        """
        return self._time_metrics(starting_points, ending_points, "idle")


    def split(
        self, seperation_events: list[str], events_preceding: int = -1, events_following: int = -1
    ) -> list[pl.DataFrame]:
        """Splitst de DataFrame op basis van de opgegeven scheidingsevents.

        Elke match levert een apart venster op; het scheidingsevent zelf is inbegrepen.

        Args:
            seperation_events (list[str]): De events waarop gesplitst moet worden.
            events_preceding (int, optional): Het aantal events dat voorafgaand aan het scheidingsevent meegenomen moet worden. Defaults to -1 (alle).
            events_following (int, optional): Het aantal events dat volgend op het scheidingsevent meegenomen moet worden. Defaults to -1 (alle).

        Returns:
            list[pl.DataFrame]: Lijst van DataFrames met de gesplitste cases.
        """  # noqa: E501
        if events_preceding < -1 or events_following < -1:
            raise ValueError("events_preceding en events_following moeten -1 of niet-negatief zijn.")

        parts = []
        for case in self._case_groups():
            activities = case.get_column(self._event_col).to_list()
            for index, event in enumerate(activities):
                if event not in seperation_events:
                    continue
                first = 0 if events_preceding == -1 else max(0, index - events_preceding)
                last = (
                    case.height - 1
                    if events_following == -1
                    else min(case.height - 1, index + events_following)
                )
                row_indices = set(case.slice(first, last - first + 1)["__pm_row_index"].to_list())
                parts.append(self._filter_rows(row_indices))
        return parts


    def unravel(self, event_types: list[str] = []) -> pl.DataFrame:
        """Maakt van de DataFrame een 'unraveled' versie, waarbij event uitgebreid wordt met een nr.

        Het nummer is de nulgebaseerde positie in de case. Het eerste event blijft ongewijzigd;
        alleen geselecteerde eventsoorten krijgen vanaf de tweede positie een suffix.

        Args:
            event_types (list(str)): Indien gevuld worden alleen die event_types unraveled. Defaults to [].

        Returns:
            pl.DataFrame: Unraveled DataFrame.
        """  # noqa: E501
        selected_types = set(event_types)
        new_values = self._df.get_column(self._event_col).to_list()
        for case in self._case_groups():
            for position, (event, row_index) in enumerate(
                zip(
                    case.get_column(self._event_col).to_list(),
                    case.get_column("__pm_row_index").to_list(),
                )
            ):
                if event is None or (selected_types and event not in selected_types):
                    continue
                if position:
                    new_values[row_index] = f"{event}_{position}"
        return self._df.with_columns(
            pl.Series(self._event_col, new_values, dtype=pl.String)
        )