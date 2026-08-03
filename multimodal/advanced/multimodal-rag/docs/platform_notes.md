# Platform notes

## Traffic path
All external traffic enters through the edge proxy and is terminated at the API
gateway. Individual services are never exposed publicly.

## Data movement
Application state is held in Postgres. Analytical copies reach the warehouse
through a change-capture pipeline, usually within ten minutes of the write.

## Reliability
Each service carries a monthly error budget. When the remaining budget crosses
the alert line, feature work stops and the team switches to reliability work
until the budget recovers. The exact weekly trajectory is tracked on the error
budget chart.
