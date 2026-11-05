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

    def bereken_doorlooptijd(self) -> pl.DataFrame:
        """Berekent de totale doorlooptijd per case."""
        return (
            self._df.group_by(self._case_col)
            .agg(
                [
                    pl.col(self._start_col).min().alias("case_start"),
                    pl.col(self._end_col).max().alias("case_end"),
                ]
            )
            .with_columns((pl.col("case_end") - pl.col("case_start")).alias("doorlooptijd"))
        )

    def aantal_cases(self) -> int:
        """Geeft het unieke aantal cases terug."""
        return self._df.select(pl.col(self._case_col).n_unique()).item()

    def agg_per_case(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggegreert de DataFrame op basis van aggregatie regels over de case_id kolom.

        Args:
            aggregation_rules (list[tuple[str, str, str]]): Lijst van tuples die de aggregatie regels definiëren
                source_column: De kolom waarop geaggregeerd moet worden
                aggregation_column: De kolom die geaggregeerd moet worden
                aggregation_function: De aggregatie functie (bijv. 'count', 'sum', 'mean')

        Returns:
            pl.DataFrame: Geaggregeerde DataFrame.
        """

    def agg_case_variable(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggegreert de case variablen.

        Per case wordt de variable 1 maal meegenomen in de aggregatie. Indien er meerdere waarden zijn voor een case wordt de eerste waarde meegenomen.

        Args:
            aggregation_rules (list[tuple[str, str, str]]): Lijst van tuples die de aggregatie regels definiëren
                source_column: De kolom waarop geaggregeerd moet worden
                aggregation_column: De kolom die geaggregeerd moet worden
                aggregation_function: De aggregatie functie (bijv. 'count', 'sum', 'mean')

        Returns:
            pl.DataFrame: Geaggregeerde DataFrame.
        """  # noqa: E501

    def throughput_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Bereken de doorlooptijd per case.

        Indien starting_points en of ending_points zijn opgegeven, wordt de doorlooptijd berekend tussen deze punten.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met case_id, case_start, case_end en throughput_time kolommen.
        """

    def processing_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Bereken de verwerkingstijd per case.

        Indien starting_points en of ending_points zijn opgegeven, wordt de verwerkingstijd berekend tussen deze punten.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met case_id, case_start, case_end en processing_time kolommen.
        """

    def idle_time(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Bereken de inactieve tijd per case.

        Indien starting_points en of ending_points zijn opgegeven, wordt de inactieve tijd berekend tussen deze punten.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met case_id, case_start, case_end en idle_time kolommen.
        """

    def filter_precedence(
        self, antecedents: list[str], consequents: list[str], inverse: bool = False, max_gap: int = -1
    ) -> pl.DataFrame:
        """Filtert cases met de opgegeven volgordes op basis van de opgegeven antecedenten en consequents.

        Args:
            antecedents (list[str]): De antecedenten waarop gefilterd moet worden.
            consequents (list[str]): De consequents waarop gefilterd moet worden.
            inverse (bool, optional): Indien True, worden de cases die niet voldoen aan de filtercriteria meegenomen. Defaults to False.
            max_gap (int, optional): Het maximale aantal events dat tussen het antecedent en consequent mag zitten. Defaults to -1 (geen limiet).

        Returns:
            pl.DataFrame: Gefilterde DataFrame met alleen de cases die voldoen aan de filtercriteria.
        """  # noqa: E501

    def trim(self, starting_points: list[str] = [], ending_points: list[str] = []) -> pl.DataFrame:
        """Trimt de DataFrame op basis van de opgegeven start- en eindpunten.

        Args:
            starting_points (list[str], optional): startpunten van de te meten procesdelen. Defaults to [].
            ending_points (list[str], optional): eindpunten van de te meten procesdelen. Defaults to [].

        Returns:
            pl.DataFrame: DataFrame met alleen de events tussen de opgegeven start- en eindpunten.
        """

    def split(
        self, seperation_events: list[str], events_preceding: int = -1, events_following: int = -1
    ) -> list[pl.DataFrame]:
        """Splitst de DataFrame op basis van de opgegeven scheidingsevents.

        Args:
            seperation_events (list[str]): De events waarop gesplitst moet worden.
            events_preceding (int, optional): Het aantal events dat voorafgaand aan het scheidingsevent meegenomen moet worden. Defaults to -1 (alle).
            events_following (int, optional): Het aantal events dat volgend op het scheidingsevent meegenomen moet worden. Defaults to -1 (alle).

        Returns:
            list[pl.DataFrame]: Lijst van DataFrames met de gesplitste cases.
        """  # noqa: E501

    def head_cases(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de eerste n of de eerste percentage van de cases terug in de DataFrame.

        Sorteren en vervolgens head of tail helpt om in te zoomen op meest voorkomede of langst lopende cases enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende cases worden geselecteerd.

        Args:
            n (int, optional): Aantal over te houden cases. Defaults to -1 (all).
            percentage (float, optional): Percentage van de cases dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Geslicede DataFrame waar alleen de cases tussen de opgegeven start- en eindindex in zitten.
        """

    def tail_cases(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de laatste n of de laatste percentage van de cases terug in de DataFrame.

        Sorteren en vervolgens head of tail helpt om in te zoomen op meest voorkomede of langst lopende cases enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende cases worden geselecteerd.

        Args:
            n (int, optional): Aantal over te houden cases. Defaults to -1 (all).
            percentage (float, optional): Percentage van de cases dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Geslicede DataFrame waar alleen de cases tussen de opgegeven start- en eindindex in zitten.
        """

    def head_event_types(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de eerste n of de eerste percentage van de eventsoorten terug in de DataFrame.

        Indien start of einde -1 zijn dan is aan die kant geen limiet en wordt alles meegenomen.
        Sorteren en vervolgens slicen helpt om in te zoomen op meest voorkomede of langst durende soorten events enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende eventsoorten worden geselecteerd.

        Args:
            n (int, optional): Aantal over te houden eventsoorten. Defaults to -1 (all).
            percentage (float, optional): Percentage van de eventsoorten dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Gefilterd dataframe.
        """  # noqa: E501

    def tail_event_types(self, n: int = -1, percentage: float = 1.0) -> pl.DataFrame:
        """Geeft de laatste n of de laatste percentage van de eventsoorten terug in de DataFrame.

        Indien start of einde -1 zijn dan is aan die kant geen limiet en wordt alles meegenomen.
        Sorteren en vervolgens slicen helpt om in te zoomen op meest voorkomede of langst durende soorten events enz.
        Door eerst te sorteren op aantal voorkomens kunnen zo de meest of minst voorkomende eventsoorten worden geselecteerd.

        Args:
            n (int, optional): Aantal over te houden eventsoorten. Defaults to -1 (all).
            percentage (float, optional): Percentage van de eventsoorten dat meegenomen moet worden. Defaults to 1.0 (alle).

        Returns:
            pl.DataFrame: Gefilterd dataframe.
        """  # noqa: E501

    def aggregate_event_types(self, aggregation_rules: list[tuple[str, str, str]]) -> pl.DataFrame:
        """Aggegreert de DataFrame op basis van aggregatie regels over de eventsoorten.

        Args:
            aggregation_rules (list[tuple[str, str, str]]): Lijst van tuples met
                source_column: De kolom waarop geaggregeerd moet worden
                aggregation_column: De kolom die geaggregeerd moet worden
                aggregation_function: De aggregatie functie (bijv. 'count', 'sum', 'mean')

        Returns:
            pl.DataFrame: Geaggregeerde DataFrame.
        """

    def unravel(self, event_types: list[str] = []) -> pl.DataFrame:
        """Maakt van de DataFrame een 'unraveled' versie, waarbij event uitgebreid wordt met een nr.

        Het nummer is gelijk aan het aantal events dat al eerder in de case heeft plaatsgevonden.
        Het eerste event dat nummer 0 zou krijgen, krijgt geen nummer, het tweede event dat plaatsvindt in de case krijgt nummer 1, enz.

        Args:
            event_types (list(str)): Indien gevuld worden alleen die event_types unraveled. Defaults to [].

        Returns:
            pl.DataFrame: Unraveled DataFrame.
        """  # noqa: E501
