import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature, threshold, impurity, samples,
                 class_distribution,
                 treeclass, is_leaf,
                 node_children):
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.samples = samples
        self.class_distribution = class_distribution
        self.treeclass = treeclass
        self.is_leaf = is_leaf
        self.node_children = node_children
        self.nodes = dict()
        self.start_end_x_axis = []
        self.node_order = ["left", "right"]


    def to_dict(self):
        def convert(value):
            if isinstance(
                value,
                (
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(value)
            if isinstance(value, (np.float16, np.float32, np.float64) + (getattr(np, 'float128', ()),)):
                return float(value)
            if isinstance(value, np.ndarray):
                return [convert(item) for item in value.tolist()]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, pd.Series):
                return [convert(item) for item in value.tolist()]
            if isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        node_dict = {
            "feature": int(self.feature) if isinstance(self.feature, np.longlong) else self.feature,
            "threshold": self.threshold,
            "impurity": convert(self.impurity),
            "samples": int(self.samples) if isinstance(self.samples, np.longlong) else self.samples,
            "class_distribution": [convert(val) for val in self.class_distribution],
            "treeclass": convert(self.treeclass),
            "is_leaf": self.is_leaf,
            "start_end_x_axis": [convert(val) for val in self.start_end_x_axis] if self.start_end_x_axis is not None else None,

        }

        node_dict["node_children"] = []
        for key in self.node_order:
            node_dict["node_children"].append(self.nodes[key].to_dict() if key in self.nodes else None)

        return node_dict


    def add_node(self, key, node):
        self.nodes[key] = node

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()
