#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Alisue (lambdalisue@hashnote.net)
# URL:      http://hashnote.net/
# Date:     2013-07-20
#
# (C) 2013 hashnote.net, Alisue
#
import re
import numpy as np


class Data(object):
    delimiter = "\s+"
    comments = "#"
    skiprows = 0
    indexer = None
    dtype=float

    def __init__(self, datalist, index=None):
        self.data = datalist
        self.index = index

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            cls.parse(f)

    @classmethod
    def parse(cls, raw):
        comment_pattern = re.compile("%s.*$" % cls.comments)
        delimiter_pattern = re.compile(cls.delimiter)
        datalist = []
        index = None

        for i, row in enumerate(raw):
            # indexing if defined
            if not index and cls.indexer:
                index = cls.indexer(cls, i, row)
                if index:
                    continue
            # skip row in skiprows
            if i < cls.skiprows:
                continue
            # remove comments
            if cls.comments:
                row = comment_pattern.sub("", row)
            # remove leading/trailing spaces
            row = row.strip()
            # skip empty row
            if len(row) == 0:
                continue
            # split row into columns
            columns = delimiter_pattern.split(row)
            datalist.append(columns)
        # transform to numpy array
        data = cls(np.array(datalist, dtype=cls.dtype), index)
        return data

    def __getitem__(self, key):
        if self.index and isinstance(key, basestring) and key in self.index:
            i = self.index[key]
            return self.data[:,i]
        return self.data[key]

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

class Indexer(object):
    def __call__(self, cls, i, row):
        # remove leading/trailing spaces
        row = row.strip()
        # split row with delimiter
        delimiter_pattern = re.compile(cls.delimiter)
        columns = delimiter_pattern.split(row)
        # create index
        index = zip(columns, range(len(columns)))
        index = dict(index)
        return index

class CommentIndexer(Indexer):
    def __call__(self, cls, i, row):
        if not row.strip().startswith(cls.comments):
            return None
        # remove comment string
        row = row.replace(cls.comments, "")
        return super(CommentIndexer, self).__call__(cls, i, row)
