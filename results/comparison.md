# SelfCheckGPT vs Ragas: comparison report

SelfCheck flags when score > 0.4; Ragas flags when faithfulness < 0.5.

![comparison](comparison.png)

## Summary

| verdict                  |   count |
|:-------------------------|--------:|
| agree: grounded          |      22 |
| DISAGREE: SelfCheck only |       1 |
| DISAGREE: Ragas only     |       1 |

## Per question

| id   | type   |   selfcheck_score |   faithfulness |   answer_relevancy |   context_precision |   context_recall | verdict                  |
|:-----|:-------|------------------:|---------------:|-------------------:|--------------------:|-----------------:|:-------------------------|
| q01  | normal |              0.01 |            1   |               0.97 |                1    |             1    | agree: grounded          |
| q02  | normal |              0    |            0.5 |               1    |                1    |             1    | agree: grounded          |
| q03  | normal |              0.09 |            1   |               0.96 |                1    |             1    | agree: grounded          |
| q04  | normal |              0    |            1   |               0.98 |                1    |             1    | agree: grounded          |
| q05  | normal |              0    |            1   |               0.99 |                1    |             1    | agree: grounded          |
| q06  | normal |              0.11 |            1   |               0.69 |                0.5  |             0.5  | agree: grounded          |
| q07  | normal |              0.06 |            1   |               0.83 |                1    |             1    | agree: grounded          |
| q08  | normal |              0.26 |            1   |               1    |                1    |             1    | agree: grounded          |
| q09  | normal |              0    |            1   |               0.98 |                1    |             0.67 | agree: grounded          |
| q10  | normal |              0    |            1   |               0.97 |                1    |             1    | agree: grounded          |
| q11  | normal |              0    |            1   |               0.75 |                1    |             0.5  | agree: grounded          |
| q12  | normal |              0.02 |            1   |             nan    |                1    |             1    | agree: grounded          |
| q13  | normal |              0    |            1   |               0.98 |                1    |             1    | agree: grounded          |
| q14  | normal |              0.41 |            1   |               0.99 |                0.83 |             0.5  | DISAGREE: SelfCheck only |
| q15  | normal |              0.08 |            1   |               0.75 |                1    |             0.5  | agree: grounded          |
| t01  | trap   |              0.09 |            0.5 |               0    |                1    |             1    | agree: grounded          |
| t02  | trap   |              0    |            0.5 |               0    |                0    |             1    | agree: grounded          |
| t03  | trap   |              0    |            1   |               0    |                0    |             1    | agree: grounded          |
| t04  | trap   |              0    |            0.5 |               0    |              nan    |             1    | agree: grounded          |
| t05  | trap   |              0    |            0.5 |             nan    |              nan    |           nan    | agree: grounded          |
| t06  | trap   |              0    |          nan   |             nan    |                0.83 |             1    | agree: grounded          |
| t07  | trap   |              0    |          nan   |             nan    |              nan    |             1    | agree: grounded          |
| t08  | trap   |              0    |            0   |             nan    |              nan    |           nan    | DISAGREE: Ragas only     |
| t09  | trap   |              0    |          nan   |             nan    |                0    |             1    | agree: grounded          |

## Disagreements

### q14 (normal): DISAGREE: SelfCheck only

**Q:** Can I get my Nimbus Phone X2 bought in Canada serviced in the United Kingdom?

**A:** No, you cannot get your Nimbus Phone X2 bought in Canada serviced in the United Kingdom. It can only be serviced in the country or region where it was originally purchased.

SelfCheck = 0.41, faithfulness = 1.00

### t08 (trap): DISAGREE: Ragas only

**Q:** How long does express shipping take to Japan?

**A:** I don't know based on the provided documents.

SelfCheck = 0.00, faithfulness = 0.00
