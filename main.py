from src.chromatic_tda.utils.timing import TimingUtils
from tests.test_bars import main as test_bars_main
from tests.timing.timing_analysis import TimingAnalysis


def main():
    test_bars_main()

    # TimingAnalysis(
    #     n=1000,
    #     color_range_splits=(.5, 1),
    #     sub_complex='monochromatic'
    # ).run()
    # TimingUtils().flush()
    #
    # TimingAnalysis(
    #     n=500,
    #     color_range_splits=(.3, .6),
    #     sub_complex='monochromatic'
    # ).run()


if __name__ == "__main__":
    main()
