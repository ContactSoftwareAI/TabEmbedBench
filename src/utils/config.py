from enum import Enum


class EmbAggregation(Enum):
    MEAN = "mean"
    CONCAT = "concat"
    PERCENTILE = "percentile"
    COLUMN = "column"
