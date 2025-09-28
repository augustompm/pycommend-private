"""
Profile MOVNS to identify performance bottlenecks
"""

import sys
import cProfile
import pstats
from io import StringIO

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')
sys.path.append('E:/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS


def profile_movns():
    """
    Profile MOVNS execution
    """
    movns = MOVNS_VNS('numpy', archive_size=10, max_iterations=1, track_metrics=False)

    profiler = cProfile.Profile()
    profiler.enable()

    movns.initialize_archive()
    neighborhoods = movns.define_neighborhoods()

    test_solution = movns.smart_initialization('medium')

    for i in range(3):
        improved = movns.mobi_p_local_search(test_solution, neighborhoods[0])

    profiler.disable()

    stream = StringIO()
    ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    ps.print_stats(20)

    print(stream.getvalue())

    stream = StringIO()
    ps = pstats.Stats(profiler, stream=stream).sort_stats('time')
    ps.print_stats(10)

    print("\nTop 10 by time:")
    print(stream.getvalue())


if __name__ == '__main__':
    profile_movns()