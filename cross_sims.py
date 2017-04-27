#!/usr/local/bin/python3

from __future__ import print_function

import scipy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import gzip


inspect = lambda x: [x][print(x) or print(type(x))]


def load_data(fn,
              zipped=False,
              line_filter=None,
              target_filter=None,
              context_filter=None):
    # if target_filter is not None:
    #     first_tc_filter = lambda t, c: target_filter(t)
    # else:
    #     first_tc_filter = lambda _, __: True
    # if context_filter is not None:
    #     tc_filter = lambda t, c: context_filter(c) and first_tc_filter(t, c)
    # else:
    #     tc_filter = first_tc_filter
    if line_filter is None:
        line_filter = lambda _: True
    
    opener = open if not zipped else lambda fn: (s.decode("utf8") for s in gzip.open(fn))
    
    try:
        lines = opener(fn)
    except OSError:
        s = fn
        lines = s.strip().splitlines()
    data = [(t, c, float(n))
            for line in lines
            if line_filter(line)
            for (t, c, n) in [line.split()]
            if (context_filter is None) or context_filter(c)
            if (target_filter is None) or target_filter(t)
            ]
    return data


def unzip(seq):
    res = tuple()
    for el in seq:
        if res:
            for (el2, place) in zip(el, res):
                place.append(el2)
        else:
            for el2 in el:
                res += ([el2],)
    return res



indexify = lambda xs: [(index_stream, index_map) for index_map, index_stream in [({}, [])]
                       for x in xs                                                                              
                       for i in [index_map.setdefault(x, len(index_map))]
                       for _ in [index_stream.append(i)]][0]


indexify_data = lambda data: [                              
    (data_, (tm, cm))                                                                                               
    for index_info in [list(map(indexify, unzip(data)[:2]))]
    for ti, tm in [index_info[0]]
    for ci, cm in [index_info[1]]
    for vals in [unzip(data)[-1]]
    for data_ in [(vals, (ti, ci))]][0]


def cross_sims(data):
    data_, (target_map, index_map) = indexify_data(data)
    data_ = (np.array(data_[0]), tuple(map(np.array, data_[1])))
    m = scipy.sparse.csr_matrix(data_, dtype=float)
    lens = scipy.sparse.linalg.norm(m, axis=1, ord=2)
    m_ = (m.transpose() / lens).transpose()
    return m_ * m_.transpose(), target_map #, index_map

def test():
    s = """                                          
t1 c1 1                                                                                                         
t1 c2 2
t1 c3 3
t2 c1 10
t2 c3 20
t3 c2 100
t3 c3 200
    """
    
    data = load_data(s)
    print(cross_sims(data))




class CrossSims:
    def __init__(self, opts):
        # opts.
        try:
            targets = set(open(opts.TARGETS).read().strip().split())
            print("targets", targets)
            target_filter = targets.__contains__
        except TypeError:
            target_filter = None
        try:
            contexts = set(open(opts.CONTEXTS).read().strip().split())
            print("contexts", contexts)
            context_filter = contexts.__contains__
        except TypeError:
            context_filter = None
        if opts.LINEFILTER == "%":
            line_filter = lambda line: line and line[:2] != "%\t"
        else:
            line_filter = None
        
        data = load_data(fn=opts.FILE,
                         zipped=opts.ZIPPED,
                         line_filter=line_filter,
                         target_filter=target_filter,
                         context_filter=context_filter)
        
        if opts.SAVESPACE:
            with open(opts.SAVESPACE, "w") as f:
                lines = ("\t".join(map(str, row)) for row in data)
                f.writelines(line+"\n" for line in lines)
        if opts.SAVESIMS:
            self.result = cross_sims(data)
            self.save_result(opts.SAVESIMS)
    
    def save_result(self, fn):
        assert hasattr(self, "result")
        matrix, target_map = self.result
        I, J = matrix.shape
        
        rev_map = dict((v, k) for k, v in target_map.items())
        sims = ((t1, t2, sim)
                 for i in range(I)
                 for j in range(i+1, J)
                 for t1 in [rev_map[i]]
                 for t2 in [rev_map[j]]
                 for sim in [matrix[i,j]]
                )
        lines = list("\t".join(map(str, sim_trip)) for sim_trip in sims)
        with open(fn, 'w') as f:
            f.writelines(line+"\n" for line in lines)



def main():
    import sys
    from optparse import OptionParser
    optparser = OptionParser()
    
    optparser.add_option("-f", "--file", dest="FILE")
    optparser.add_option("-t", "--targets", dest="TARGETS", help="file name")
    optparser.add_option("-c", "--contexts", dest="CONTEXTS", help="file name")
    optparser.add_option("-l", "--line-filter", dest="LINEFILTER", action="store_true")
    optparser.add_option("-z", "--zipped", dest="ZIPPED", action="store_true")
    optparser.add_option("-s", "--save-space", dest="SAVESPACE", help="file name")
    optparser.add_option("-x", "--save-sims", dest="SAVESIMS", help="file name")
    
    optparser.add_option("--test", dest="TEST", action="store_true")
    
    opts, args = optparser.parse_args()
    if opts.TEST:
        test()
        return
    
    crosssims = CrossSims(opts)

if __name__ == '__main__':
    main()

