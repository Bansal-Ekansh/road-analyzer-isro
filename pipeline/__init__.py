def __getattr__(name):
    if name == "RoadSegmenter":
        from .segmentation import RoadSegmenter
        return RoadSegmenter
    if name == "GraphBuilder":
        from .graph_builder import GraphBuilder
        return GraphBuilder
    if name == "GraphHealer":
        from .healing import GraphHealer
        return GraphHealer
    if name == "GraphAnalyzer":
        from .analytics import GraphAnalyzer
        return GraphAnalyzer
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
