from __future__ import annotations

from datetime import timedelta
from math import ceil

import polars as pl


@pl.api.register_dataframe_namespace("pm")
class processminerSVB:
    def __init__(self, df: pl.DataFrame):
        """Initialize the namespace with a DataFrame.

        Args:
            df (pl.DataFrame): DataFrame to expose through the process-mining namespace.

        Returns:
            None: This method initializes the namespace instance.
        """
        self._df = df

        # Default column names
        self._case_col = "case_id"
        self._event_col = "activity"
        self._start_col = "start_time"
        self._end_col = "end_time"

# Internals #################################################################################

    def _case_groups(self) -> list[pl.DataFrame]:
        """Return a list of DataFrames, each containing a case's rows in their original event-log order.

        Returns:
            list[pl.DataFrame]: List of DataFrames, each containing a case's rows in their original event-log order.
        """
        indexed = self._df.with_row_index("__pm_row_index")
        return indexed.partition_by(self._case_col, maintain_order=True)


    def _filter_rows_by_indices(self, row_indices: set[int]) -> pl.DataFrame:
        """Return the original rows with the selected row indices.

        Args:
            row_indices (set[int]): Row indices to retain.

        Returns:
            pl.DataFrame: DataFrame containing the selected original rows.
        """
        return self._df.filter(
            pl.Series(
                "__pm_selected",
                [index in row_indices for index in range(self._df.height)],
                dtype=pl.Boolean,
            )
        )


    def _filter_values(self, column: str, values: list[object]) -> pl.DataFrame:
        """Filter rows by values present in a column.

        Args:
            column (str): Name of the column to inspect.
            values (list[object]): Values to retain, including None to select nulls.

        Returns:
            pl.DataFrame: DataFrame containing rows that match the selected values.
        """
        selected_values = [value for value in values if value is not None]
        predicate = pl.col(column).is_in(selected_values)
        if any(value is None for value in values):
            predicate = predicate | pl.col(column).is_null()
        return self._df.filter(predicate)



    def _calculate_amount(self, n: int, percentage: float, total: int) -> int:
        """Determine how many ranked items to select.

        Args:
            n (int): Maximum number of items to select, or -1 to use the percentage.
            percentage (float): Fraction of the total to select, between 0 and 1.
            total (int): Total number of available items.

        Returns:
            int: Number of items to select.
        """
        if n < -1:
            raise ValueError("n must be -1 (all) or a non-negative number.")
        if not 0 <= percentage <= 1:
            raise ValueError("percentage must be between 0 and 1.")
        percentage_count = ceil(total * percentage)
        return percentage_count if n == -1 else min(n, percentage_count)


    def _ranked_cases(self, descending: bool) -> list[tuple[object, int]]:
        """Rank cases by their number of events.

        Args:
            descending (bool): Whether to sort from the most events to the fewest.

        Returns:
            list[tuple[object, int]]: Case identifiers paired with event counts, in rank order.
        """
        ranked: list[tuple[object, int]] = []
        for case in self._case_groups():
            case_id = case[self._case_col][0]
            ranked.append((case_id, case.height))
        return sorted(ranked, key=lambda item: item[1], reverse=descending)


    def _ranked_event_types(self, descending: bool) -> list[tuple[object, int]]:
        """Rank event types by their occurrence count.

        Args:
            descending (bool): Whether to sort from the most occurrences to the fewest.

        Returns:
            list[tuple[object, int]]: Event types paired with occurrence counts, in rank order.
        """
        counts: dict[object, int] = {}
        for event in self._df.get_column(self._event_col).to_list():
            counts[event] = counts.get(event, 0) + 1
        ranked = list(counts.items())
        return sorted(ranked, key=lambda item: item[1], reverse=descending)


    def _aggregate(
        self, aggregation_rules: list[tuple[str, str, str]], group_column: str
    ) -> pl.DataFrame:
        """Aggregate the DataFrame using the specified rules.

        Args:
            aggregation_rules (list[tuple[str, str, str]]): Source column, output column,
                and aggregation function for each aggregation.
            group_column (str): Column whose values define the groups.

        Returns:
            pl.DataFrame: DataFrame containing the grouped aggregation results.
        """
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
                raise ValueError(f"Column '{source_column}' does not exist in the DataFrame.")
            if function not in aggregations:
                supported = ", ".join(aggregations)
                raise ValueError(
                    f"Unknown aggregation function '{function}'. Choose from: {supported}."
                )
            expressions.append(aggregations[function](pl.col(source_column)).alias(output_column))
        if not expressions:
            return self._df.select(group_column).unique(maintain_order=True)
        return self._df.group_by(group_column, maintain_order=True).agg(expressions)
# @@@ tot hier gecontroleerd

    def _time_window(
        self, starting_points: list[str], ending_points: list[str]
    ) -> list[tuple[pl.DataFrame, int, int]]:
        """Return inclusive row-order windows between the selected event boundaries.

        Args:
            starting_points (list[str]): Event types that can start a window. An empty list
                uses the first event in each case.
            ending_points (list[str]): Event types that can end a window. An empty list
                uses the last event in each case.

        Returns:
            list[tuple[pl.DataFrame, int, int]]: Case DataFrames paired with the first and
                last row positions of each selected window.
        """
        if (starting_points or ending_points) and self._event_col not in self._df.columns:
            raise ValueError(
                f"Column '{self._event_col}' is required to select start or end points."
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
        """Calculate a time metric for each selected case window.

        Args:
            starting_points (list[str]): Event types that can start a window.
            ending_points (list[str]): Event types that can end a window.
            metric (str): Metric to calculate: throughput, processing, or idle time.

        Returns:
            pl.DataFrame: Per-case start and end times and the calculated duration.
        """
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
        """Allow methods of the underlying DataFrame to be called via the namespace wrapper.

        Args:
            name (str): The name of the method to call.

        Returns:
            object: The requested DataFrame attribute or a wrapper for its callable.
        """
        # Look up attributes not defined by the wrapper on the underlying DataFrame.
        attribute = getattr(self._df, name)
        if not callable(attribute):
            return attribute

        # Forward callable attributes to the underlying DataFrame.
        def delegated(*args, **kwargs):
            """Call the Polars method and wrap DataFrame results in the pm namespace.

            Args:
                *args: Positional arguments passed to the Polars method.
                **kwargs: Keyword arguments passed to the Polars method.

            Returns:
                object: The Polars method result, wrapped in the namespace if it is a DataFrame.
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
        """Validate column names and assign their meanings.

        Args:
            case_id (str, optional): Name of the case identifier column. Defaults to None.
            event_id (str, optional): Name of the event type column. Defaults to None.
            start_time (str, optional): Name of the event start-time column. Defaults to None.
            end_time (str, optional): Name of the event end-time column. Defaults to None.

        Returns:
            processminerSVB: This namespace instance, allowing method chaining.
        """
        colomn_names = self._df.columns

        if case_id:
            if case_id not in colomn_names:
                raise ValueError(f"case_id column '{case_id}' does not exist in the DataFrame.")
            self._case_col = case_id
        if event_id:
            if event_id not in colomn_names:
                raise ValueError(f"event_id column '{event_id}' does not exist in the DataFrame.")
            self._event_col = event_id
        if start_time:
            if start_time not in colomn_names:
                raise ValueError(f"start_time column '{start_time}' does not exist in the DataFrame.")
            self._start_col = start_time
        if end_time:
            if end_time not in colomn_names:
                raise ValueError(f"end_time column '{end_time}' does not exist in the DataFrame.")
            self._end_col = end_time
        return self


# Filters ###########################################################################################

    def filter_precedence(
        self, antecedents: list[str], consequents: list[str], inverse: bool = False, max_gap: int = -1
    ) -> pl.DataFrame:
        """Filter cases according to the specified antecedents and consequents.

        A case matches if at least one antecedent is followed by a consequent.
        max_gap counts only events strictly between those two events.

        Args:
            antecedents (list[str]): Antecedents to filter by.
            consequents (list[str]): Consequents to filter by.
            inverse (bool, optional): If True, include cases that do not match the criteria. Defaults to False.
            max_gap (int, optional): Maximum number of events allowed between the antecedent and consequent. Defaults to -1 (unlimited).

        Returns:
            pl.DataFrame: Filtered DataFrame containing matching cases.
        """  # noqa: E501
        if not antecedents or not consequents:
            raise ValueError("antecedents and consequents must not be empty.")
        if max_gap < -1:
            raise ValueError("max_gap must be -1 (unlimited) or a non-negative number.")

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
        return self._filter_rows_by_indices(matching_rows)

    def trim(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Trim the DataFrame to the specified start and end points.

        For each case, use the first matching start and last matching end activity.
        Boundary events are retained; an unspecified boundary uses the case boundary.

        Args:
            starting_points (list[str], optional): Start points of the process segments to measure. Defaults to [].
            ending_points (list[str], optional): End points of the process segments to measure. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame containing only events between the specified start and end points.
        """
        selected: set[int] = set()
        for case, first, last in self._time_window(starting_points, ending_points):
            selected.update(case.slice(first, last - first + 1).get_column("__pm_row_index").to_list())
        return self._filter_rows_by_indices(selected)


    def head_cases(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Return the first n cases or the specified percentage of cases.

        Sorting before taking the head or tail helps focus on the most frequent or longest-running cases.
        Sorting by event count allows the most or least frequent cases to be selected.
        head_cases sorts cases by event count in descending order.

        Args:
            n (int, optional): Number of cases to keep. Defaults to -1 (all).
            percentage (float, optional): Percentage of cases to include. Defaults to 1.0 (all).

        Returns:
            pl.DataFrame: Sliced DataFrame containing only the selected cases.
        """
        ranked = self._ranked_cases(descending=True)
        count = self._calculate_amount(n, percentage, len(ranked))
        return self._filter_values(self._case_col, [case_id for case_id, _ in ranked[:count]])

    def tail_cases(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Return the last n cases or the specified percentage of cases.

        Sorting before taking the head or tail helps focus on the most frequent or longest-running cases.
        Sorting by event count allows the most or least frequent cases to be selected.
        tail_cases sorts cases by event count in ascending order.

        Args:
            n (int, optional): Number of cases to keep. Defaults to -1 (all).
            percentage (float, optional): Percentage of cases to include. Defaults to 1.0 (all).

        Returns:
            pl.DataFrame: Sliced DataFrame containing only the selected cases.
        """
        ranked = self._ranked_cases(descending=False)
        count = self._calculate_amount(n, percentage, len(ranked))
        return self._filter_values(self._case_col, [case_id for case_id, _ in ranked[:count]])

    def head_event_types(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Return the first n event types or the specified percentage of event types.

        Event types are sorted by occurrence count in descending order.

        If n is -1, there is no limit and all event types are included.
        Sorting before slicing helps focus on the most frequent or longest-running event types.
        Sorting by occurrence count allows the most or least frequent event types to be selected.

        Args:
            n (int, optional): Number of event types to keep. Defaults to -1 (all).
            percentage (float, optional): Percentage of event types to include. Defaults to 1.0 (all).

        Returns:
            pl.DataFrame: Filtered DataFrame.
        """  # noqa: E501
        ranked = self._ranked_event_types(descending=True)
        count = self._calculate_amount(n, percentage, len(ranked))
        selected = [event for event, _ in ranked[:count]]
        return self._filter_values(self._event_col, selected)

    def tail_event_types(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Return the last n event types or the specified percentage of event types.

        Event types are sorted by occurrence count in ascending order.

        If n is -1, there is no limit and all event types are included.
        Sorting before slicing helps focus on the most frequent or longest-running event types.
        Sorting by occurrence count allows the most or least frequent event types to be selected.

        Args:
            n (int, optional): Number of event types to keep. Defaults to -1 (all).
            percentage (float, optional): Percentage of event types to include. Defaults to 1.0 (all).

        Returns:
            pl.DataFrame: Filtered DataFrame.
        """  # noqa: E501
        ranked = self._ranked_event_types(descending=False)
        count = self._calculate_amount(n, percentage, len(ranked))
        selected = [event for event, _ in ranked[:count]]
        return self._filter_values(self._event_col, selected)


# Enrichment ############################################################################################

    def troughput_time(self) -> pl.DataFrame:
        """Calculate the total throughput time per case.

        Returns:
            pl.DataFrame: Per-case start and end times and total throughput time.
        """
        return self._time_metrics([], [], "throughput")

    def case_count(self) -> int:
        """Return the number of unique cases.

        Returns:
            int: Number of unique case identifiers in the DataFrame.
        """
        return self._df.select(pl.col(self._case_col).n_unique()).item()

    def agg_per_case(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggregate the DataFrame by case using the specified aggregation rules.

        Each rule is (source column, output column, function), for example
        ("amount", "total_amount", "sum").

        Args:
            aggregation_rules (list[tuple[str, str, str]]): Source column, output column,
                and aggregation function for each aggregation.

        Returns:
            pl.DataFrame: Aggregated DataFrame.
        """
        return self._aggregate(aggregation_rules, self._case_col)


    def agg_per_event_type(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggregate the DataFrame by event type using the specified aggregation rules.

        Each rule is (source column, output column, function), for example
        ("amount", "average_amount", "mean").

        Args:
            aggregation_rules (list[tuple[str, str, str]]): Source column, output column,
                and aggregation function for each aggregation.

        Returns:
            pl.DataFrame: Aggregated DataFrame.
        """  # noqa: E501
        return self._aggregate(aggregation_rules, self._event_col)



    def throughput_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Calculate the throughput time per case.

        If starting_points and/or ending_points are specified, calculate throughput time between those points.
        Unspecified boundaries default to the first or last event of the case, respectively.

        Args:
            starting_points (list[str], optional): Start points of the process segments to
                measure. Defaults to [].
            ending_points (list[str], optional): End points of the process segments to
                measure. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame with case_id, case_start, case_end, and throughput_time columns.
        """
        return self._time_metrics(starting_points, ending_points, "throughput")


    def processing_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Calculate the processing time per case.

        Processing time is the sum of end_time - start_time for each event in the selected case window.

        Args:
            starting_points (list[str], optional): Start points of the process segments to
                measure. Defaults to [].
            ending_points (list[str], optional): End points of the process segments to
                measure. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame with case_id, case_start, case_end, and processing_time columns.
        """
        return self._time_metrics(starting_points, ending_points, "processing")


    def idle_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Calculate the idle time per case.

        Idle time is the sum of positive gaps between an event end and the next event start
        within the selected case window.

        Args:
            starting_points (list[str], optional): Start points of the process segments to
                measure. Defaults to [].
            ending_points (list[str], optional): End points of the process segments to
                measure. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame with case_id, case_start, case_end, and idle_time columns.
        """
        return self._time_metrics(starting_points, ending_points, "idle")


    def split(
        self, seperation_events: list[str], events_preceding: int = -1, events_following: int = -1
    ) -> list[pl.DataFrame]:
        """Split the DataFrame based on the specified separator events.

        Each match produces a separate window; the separator event itself is included.

        Args:
            seperation_events (list[str]): Events on which to split.
            events_preceding (int, optional): Number of events to include before the separator event. Defaults to -1 (all).
            events_following (int, optional): Number of events to include after the separator event. Defaults to -1 (all).

        Returns:
            list[pl.DataFrame]: List of DataFrames containing the split cases.
        """  # noqa: E501
        if events_preceding < -1 or events_following < -1:
            raise ValueError("events_preceding and events_following must be -1 or non-negative.")

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
                parts.append(self._filter_rows_by_indices(row_indices))
        return parts


    def unravel(self, event_types: list[str] = []) -> pl.DataFrame:
        """Create an unraveled version of the DataFrame, suffixing events with their position.

        The number is the zero-based position in the case. The first event remains unchanged;
        only selected event types receive a suffix, starting from the second position.

        Args:
            event_types (list[str]): If provided, only these event types are unraveled. Defaults to [].

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