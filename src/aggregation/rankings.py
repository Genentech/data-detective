import typing
from enum import Enum

import pandas as pd
import scipy
from typing import List

from pyrankagg.rankagg import FullListRankAggregator


class RankingAggregationMethod(Enum):
    MEDIAN_AGGREGATION = "median_aggregation"
    HIGHEST_RANK = "highest_rank"
    LOWEST_RANK = "lowest_rank"
    STABILITY_SELECTION = "stability_selection"
    EXPONENTIAL_WEIGHTING = "exponential_weighting"
    STABILITY_ENHANCED_BORDA = "stability_enhanced_borda"
    EXPONENTIAL_ENHANCED_BORDA = "exponential_enhanced_borda"
    ROBUST_AGGREGATION = "robust_aggregation"
    ROUND_ROBIN = "round_robin"


class RankingAggregator:
    FLRA = FullListRankAggregator()

    def __init__(self, results_object):
        self.results_object = results_object

    @staticmethod
    def list_is_full_ranking(lst: List) -> bool:
        """
        Checks to see if the list contains every number between 0 and len(lst) - 1.

        @param lst: the list to check
        @return: a boolean of whether the list contains every number between 0 and len(lst) - 1.
        """
        lst_len = len(lst)
        return list(range(lst_len)) == sorted(lst)

    @staticmethod
    def convert_to_scorelist(dataframe: pd.DataFrame) -> typing.List[typing.Dict[str, float]]:
        """
        Converts a dataframe to a list of score dictionaries. Each column of the dataframe should be an
        "expert" that is giving their rankings or evaluations, and each row should represent
        the object that is being ranked or evaluated.

        Example of a scorelist:
        scorelist = [{'milk':1.4,'cheese':2.6,'eggs':1.2,'bread':3.0},
                         {'milk':2.0,'cheese':3.2,'eggs':2.7,'bread':2.9},
                         {'milk':2.7,'cheese':3.0,'eggs':2.5,'bread':3.5}]

        @param dataframe: the dataframe as described above
        @return: a list of score dictionaries, where each dictionary represents a single expert's evaluation
        of each object or sample
        """

        scorelist = []
        for col in dataframe.columns:
            tmp_dict = {f"item {idx}": val for idx, val in zip(dataframe.index, dataframe[col])}
            scorelist.append(tmp_dict)
        return scorelist

    @staticmethod
    def get_rankings(scores: list) -> typing.Dict[str, float]:
        """
        Accepts an input list in which the values are scores, in which a higher score is better.
        Returns a dictionary of items and ranks, ranks in the range 1,...,n.

        @param list: a list of the scores to be converted to rankings.
        @return a dict keying from item number to the rankings, where high score is better and higher ranking is better.
        """
        return {f"item {k}": v for k, v in RankingAggregator.FLRA.convert_to_ranks(dict(enumerate(scores))).items()}

    def construct_rankings_df(self, validator_name, given_validator_method: str = None,
                              given_data_modality: str = None) -> pd.DataFrame:
        """
        Constructs a rankings dataframe for a particular validator from the results object stored as an instance variable.

        @param validator_name: the name of the validator to find the rankings of
        @param given_validator_method: if rankings are only desired for a single validator method, users can specify
        the name of the validator method in this kwarg, and the method will only give the ranking df for that method
        @param given_data_modality: if rankings are only desired for a given data modality (this is basically just a
        diff way of saying column name), users can specify that column in this kwarg and the method will only give the
        rankings for that data modality
        @return: a rankings dataframe where the rows are items and the columns are the methods / modalitiese
        """
        validator_results = self.results_object[validator_name]
        results_obj = {}

        for validator_method, results_dict in validator_results.items():
            if given_validator_method and (validator_method != given_validator_method):
                continue
            for data_modality, scores in results_dict.items():
                if given_data_modality and (data_modality.replace("_results", "") != given_data_modality):
                    continue
                rankings = RankingAggregator.get_rankings(scores)
                results_obj[f"{data_modality}_{validator_method}_rank"] = rankings

        rankings_df = pd.DataFrame(results_obj)
        return rankings_df.sort_index()

    def aggregate_modal_rankings(self, validator_name: str, aggregation_methods: List[RankingAggregationMethod],
                                 given_data_modality: str = None) -> pd.DataFrame:
        """
        Aggregates rankings for a single column/modality using pyrankagg's rank aggregation methods.

        @param validator_name: the name of the validator to aggregate the rankings of
        @param aggregation_methods: the method to use for rank aggregation n
        @param given_data_modality: if rankings are only desired for a given data modality (this is basically just a
        diff way of saying column name), users can specify that column in this kwarg and the method will only give the
        rankings for that data modality
        @return: the dataframe with the original ranks as well as the aggregated ranks
        """
        rankings_df = self.construct_rankings_df(validator_name, given_data_modality=given_data_modality)
        output_df = rankings_df.copy()

        for aggregation_method in aggregation_methods:
            aggregation_method_name = aggregation_method.value
            scorelist = self.convert_to_scorelist(rankings_df)
            agg_method = getattr(RankingAggregator.FLRA, aggregation_method_name)
            agg_rankings = agg_method(scorelist)[1]
            output_df[f"{aggregation_method_name}_agg_rank"] = list(agg_rankings.values())

        return output_df

    def aggregate_rankings(self, validator_name: str, aggregation_methods: List[RankingAggregationMethod]) -> pd.DataFrame:
        """
        Aggregates rankings for a single validator using pyrankagg's rank aggregation methods.

        @param validator_name: the name of the validator to aggregate the rankings of
        @param aggregation_methods: the method to use for rank aggregation
        @return: the dataframe with the original ranks as well as the aggregated ranks
        """
        rankings_df = self.construct_rankings_df(validator_name)
        output_df = rankings_df.copy()

        for aggregation_method in aggregation_methods:
            aggregation_method_name = aggregation_method.value
            scorelist = self.convert_to_scorelist(rankings_df)
            agg_method = getattr(RankingAggregator.FLRA, aggregation_method_name)
            agg_rankings = agg_method(scorelist)[1]

            output_df[f"{aggregation_method_name}_agg_rank"] = list(agg_rankings.values())

        return output_df