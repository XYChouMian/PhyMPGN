
def periodic_padding(features, source_index, target_index):
    """

    Args:
        features (torch.Tensor): shape [n, ...], the origin features
        source_index (torch.Tensor): shape [m,]
        target_index (torch.Tensor): shape [m,]

    Returns:
        features (torch.Tensor): shape [n, ...], the padded features
    """
    features[target_index] = features[source_index]
    return features


def dirichlet_padding(features, padding_index, padding_value):
    """
    Dirichlet 边界条件赋值，支持时变和时不变边界

    形状组合：
        features         padding_value     处理方式
        [n, d]           [m, d]            直接赋值（时不变，单步）
        [n, t, d]        [m, d]            广播到时间维（时不变，序列）
        [n, t, d]        [m, t, d]         直接赋值（时变，序列）
    """
    if len(features.shape) == 3:
        # features: (n, t, d)
        if len(padding_value.shape) == 3:
            # padding_value: (m, t, d) - 时变边界条件
            features[padding_index] = padding_value
        else:
            # padding_value: (m, d) - 时不变边界条件，广播到时间维
            features[padding_index] = padding_value.unsqueeze(1) \
                .repeat(1, features.shape[1], 1)
    else:  # features.shape == 2, (n, d)
        # 单步更新：padding_value 必须是 (m, d)
        features[padding_index] = padding_value
    return features


def dirichlet_padding0(features, padding_index, padding_value):
    if len(features.shape) == 3:
        #  (m, t, d)
        features[padding_index] = padding_value.unsqueeze(1)\
            .repeat(1, features.shape[1], 1)
    else:  # == 2
        features[padding_index] = padding_value
    return features


def neumann_padding(features, source_index, target_index):
    features[target_index] = features[source_index]
    return features


def graph_padding(graph, clone=False):
    # print("=" * 50)
    # print(graph.y.shape)
    # print(graph.inlet_index.shape)
    # print(graph.inlet_value.shape)
    # raise

    if hasattr(graph, 'dirichlet_index'):
        graph.y = dirichlet_padding(graph.y, graph.dirichlet_index,
                                    graph.dirichlet_value)
    if hasattr(graph, 'inlet_index'):
        if hasattr(graph, 'inlet_value1'):
            graph.y = dirichlet_padding(graph.y, graph.inlet_index,
                                        graph.inlet_value1)
        else:
            graph.y = dirichlet_padding(graph.y, graph.inlet_index,
                                        graph.inlet_value)
    if hasattr(graph, 'periodic_src_index'):
        graph.y = periodic_padding(graph.y, graph.periodic_src_index,
                                   graph.periodic_tgt_index)
    if hasattr(graph, 'neumann_src_index'):
        graph.y = neumann_padding(graph.y, graph.neumann_src_index,
                                  graph.neumann_tgt_index)

    if clone:
        graph.y = graph.y.clone()


def h_padding(h, graph):
    if hasattr(graph, 'dirichlet_index'):
        h = dirichlet_padding(h, graph.dirichlet_index,
                              graph.dirichlet_h_value)
    if hasattr(graph, 'inlet_index'):
        h = dirichlet_padding(h, graph.inlet_index,
                              graph.inlet_h_value)
    if hasattr(graph, 'periodic_src_index'):
        h = periodic_padding(h, graph.periodic_src_index,
                             graph.periodic_tgt_index)
    if hasattr(graph, 'neumann_src_index'):
        h = neumann_padding(h, graph.neumann_src_index,
                            graph.neumann_tgt_index)
    return h
