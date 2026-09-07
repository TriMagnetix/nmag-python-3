# Scheduling API

Schedules describe save and action times during relaxation. Prefer the modern
identifier-first form, such as `every("step", 10)`.

::: when.when.at

::: when.when.every

## Never

`when.never` is a reusable condition that never matches. It is useful when a
schedule is assembled programmatically and an action needs to be disabled.

::: when.when.When
    options:
      members:
        - match_time
        - next_time
