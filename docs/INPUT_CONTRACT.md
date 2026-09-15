# Input Contract

The public runner expects a local Excel workbook supplied with `--workbook`.

## Supported file types

- `.xlsx`
- `.xlsm`

The workbook is read with `openpyxl` through pandas. The authentic historical student diary workbook is not included in this repository.

## Participant-sheet selection

By default, only sheets matching this regular expression are parsed:

```text
^[ABC]_\d+$
```

Examples that match:

```text
A_1
B_12
C_7
```

Other sheets are ignored unless `PipelineConfig.sheet_regex` is changed.

## Row offset

The default parser skips the first five rows of each matching sheet:

```python
skip_rows = 5
```

This is configurable.

## Weekday blocks

The reconstructed workbook layout uses repeated four-column blocks at zero-based positions:

| Day | Column positions |
| --- | --- |
| Monday | 2–5 |
| Tuesday | 6–9 |
| Wednesday | 10–13 |
| Thursday | 14–17 |
| Friday | 18–21 |
| Saturday | 22–25 |
| Sunday | 26–29 |

Each available block is interpreted as:

```text
Time | Activity | Who | Satisfaction
```

The reconstructed public parser includes a `Where` field in its normalized output schema, but the four-column layout available to this implementation does not provide a separate source column for it. `Where` is therefore left missing rather than fabricated.

## Time handling

Accepted clock representations include:

- Excel fractional-day numeric values between 0 and 1;
- Python `datetime.time` values;
- Python/pandas datetime or timestamp values;
- strings that pandas can parse as times/datetimes.

Times are normalized to a fixed reference date because only within-day ordering and durations are required.

Unparseable timestamps and rows without activity text are removed.

## Context continuation

`Activity`, `Who`, and `Satisfaction` cells are forward-filled within each weekday block to reflect visually continued cells in the reconstructed workbook structure. Timestamps are never forward-filled.

## Event duration reconstruction

Within each participant/day:

1. events are sorted by normalized start time;
2. an event ends when the next event starts;
3. the final event receives `default_last_event_minutes`, which defaults to 30 minutes.

Rows that would produce a negative duration are rejected rather than silently retained.

## Satisfaction / mood values

The runner converts `Satisfaction` to numeric with invalid values mapped to missing values. This repository does not assert a universal psychological scale definition beyond the source diary field itself; interpretation should therefore remain descriptive unless the original study documentation establishes more.
