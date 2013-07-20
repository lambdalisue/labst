#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Alisue (lambdalisue@hashnote.net)
# URL:      http://hashnote.net/
# Date:     2013-07-20
#
# (C) 2013 hashnote.net, Alisue
#
from labst.data import Data
from labst.data import CommentIndexer


class DistributionData(Data):
    indexer = CommentIndexer()


if __name__ == '__main__':
    import numpy as np
    DATA = """
    # foo bar hoge
    0 1 2
    0 1 2
    1 2 3
    1 2 3
    """
    CORRECT = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 3],
            [1, 2, 3],
        ])
    data = DistributionData.parse(DATA.split("\n"))

    assert np.array_equal(data, CORRECT)
    assert np.array_equal(data['foo'], np.array([0, 0, 1, 1]))
