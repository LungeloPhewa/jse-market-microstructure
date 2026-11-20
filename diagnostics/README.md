# Development diagnostics

Scripts written while debugging the pipeline: chunk-boundary checks for the
external sort, the merge-order violation that motivated `verify_sorted`, the
late-event inspection that exposed the two-hour timestamp shift, and depth
checks for the resilience analysis. They are kept for transparency, but they are not part of the
pipeline and some still contain the absolute paths of the original
development machine. To run one, change its path constants to `jsemicro.config.PATHS`.
