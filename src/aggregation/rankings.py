import typing
from enum import Enum

import pandas as pd
from typing import List, Union

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

class ScoreAggregationMethod(Enum):
    NORMALIZED_AVERAGE = "normalized_average"

class ScoreAggregator(): 
    def normalized_average(self, score_df): 
        norm_score_df = (score_df - score_df.min()) / (score_df.max() - score_df.min())
        return norm_score_df.mean(axis=1)

class ResultAggregator:
    FLRA = FullListRankAggregator()
    SCORE_AGGREGATOR = ScoreAggregator()

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
        return ResultAggregator.FLRA.convert_to_ranks(scores)

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
        validator_results = self.results_object[validator_name].copy()
        results_obj = {}

        for validator_method, results_dict in validator_results.items():
            if given_validator_method and (validator_method != given_validator_method):
                continue
            for data_modality, scores_or_tuple in results_dict.items():
                if given_data_modality and (data_modality.replace("_results", "") != given_data_modality):
                    continue

                # handles case where model is returned first.
                if isinstance(scores_or_tuple, tuple): 
                    scores = scores_or_tuple[0]
                else: 
                    scores = scores_or_tuple

                rankings = ResultAggregator.get_rankings(scores)
                # rankings = ResultAggregator.get_rankings(scores_or_tuple)

                results_obj[f"{data_modality}_{validator_method}_rank"] = rankings

        rankings_df = pd.DataFrame(results_obj)
        return rankings_df.sort_index()

    def construct_score_df(self, validator_name, given_validator_method: str = None,
                              given_data_modality: str = None) -> pd.DataFrame:
        validator_results = self.results_object[validator_name]
        results_obj = {}

        for validator_method, tuple_or_results_dict in validator_results.items():
            results_dict = tuple_or_results_dict if isinstance(tuple_or_results_dict, dict) else tuple_or_results_dict[0]
            if given_validator_method and (validator_method != given_validator_method):
                continue
            for data_modality, scores_or_tuple in results_dict.items():
                
                if given_data_modality and (data_modality.replace("_results", "") != given_data_modality):
                    continue

                # handles case where model is returned first.
                if isinstance(scores_or_tuple, tuple): 
                    scores = scores_or_tuple[0]
                else: 
                    scores = scores_or_tuple

                results_obj[f"{data_modality}_{validator_method}_score"] = scores

        score_df = pd.DataFrame(results_obj)
        return score_df.sort_index()

    def aggregate_results_modally(self, validator_name: str, aggregation_methods: List[Union[RankingAggregationMethod, ScoreAggregationMethod]],
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
        rankings_output_df = rankings_df.copy()

        score_df = self.construct_score_df(validator_name, given_data_modality=given_data_modality)
        score_output_df = score_df.copy()
        
        for aggregation_method in aggregation_methods:
            if aggregation_method in RankingAggregationMethod.__members__.values():
                aggregation_method_name = aggregation_method.value
                scorelist = self.convert_to_scorelist(rankings_df)
                agg_method = getattr(ResultAggregator.FLRA, aggregation_method_name)
                import pdb; pdb.set_trace()
                agg_rankings = agg_method(scorelist)[1]

                rankings_output_df[f"{aggregation_method_name}_agg_rank"] = list(agg_rankings.values())
            elif aggregation_method in ScoreAggregationMethod.__members__.values():
                aggregation_method_name = aggregation_method.value
                agg_method = getattr(ResultAggregator.SCORE_AGGREGATOR, aggregation_method_name)
                agg_scores = agg_method(score_df)
                score_output_df[f"{aggregation_method_name}_agg_score"] = agg_scores
            else:
                raise Exception(f"aggregation method {aggregation_method.value} not found in score or ranking aggregation methods. Please check in src/aggregation/rankings.py to make sure that it exists.")

        return pd.concat([rankings_output_df, score_output_df], axis=1)

    def aggregate_results_multimodally(self, validator_name: str, aggregation_methods: List[RankingAggregationMethod]) -> pd.DataFrame:
        """
        Aggregates rankings for a single validator using pyrankagg's rank aggregation methods.

        @param validator_name: the name of the validator to aggregate the rankings of
        @param aggregation_methods: the method to use for rank aggregation
        @return: the dataframe with the original ranks as well as the aggregated ranks
        """
        rankings_df = self.construct_rankings_df(validator_name)
        rankings_output_df = rankings_df.copy()

        score_df = self.construct_score_df(validator_name)
        score_output_df = score_df.copy()
        
        for aggregation_method in aggregation_methods:
            if aggregation_method in RankingAggregationMethod.__members__.values():
                aggregation_method_name = aggregation_method.value
                scorelist = self.convert_to_scorelist(rankings_df)
                agg_method = getattr(ResultAggregator.FLRA, aggregation_method_name)
                agg_rankings = agg_method(scorelist)[1]

                rankings_output_df[f"{aggregation_method_name}_agg_rank"] = list(agg_rankings.values())
            elif aggregation_method in ScoreAggregationMethod.__members__.values():
                aggregation_method_name = aggregation_method.value
                agg_method = getattr(ResultAggregator.SCORE_AGGREGATOR, aggregation_method_name)
                agg_scores = agg_method(score_df)
                score_output_df[f"{aggregation_method_name}_agg_score"] = agg_scores
            else:
                raise Exception(f"aggregation method {aggregation_method.value} not found in score or ranking aggregation methods. Please check in src/aggregation/rankings.py to make sure that it exists.")

        return pd.concat([rankings_output_df, score_output_df], axis=1)