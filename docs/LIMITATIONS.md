# Limitations (draft)

Numbers are from the pinned StereoSet and BERT revisions (CHANGE-03). The 3,857 rows are split by group
into a dev set (3,191 rows) and a held-out test set (666 rows, 17.3%). "Dev CV" numbers are means over
5-fold grouped cross-validation on the dev set; "test" numbers come from one fit on all dev rows, scored
once on the test set. The random forest is the chosen three-class configuration (`max_depth=10`,
500 trees). Full tables: `results/summary.md`.

## Neutral context confound

Every neutral row comes from StereoSet's *unrelated* sentence, which is often nonsensical (*"Ethiopian
men are floating"*). A model could learn to spot the odd sentence instead of the word. To test this we
trained the same forest on a context vector that leaves the word out. In dev CV that context-only model
reaches a neutral F1 of 0.690 (precision 0.653, recall 0.742). It is near chance on innate (0.475) and
derived (0.297), and its macro-F1 is 0.487 ± 0.082. The full forest reaches a neutral F1 of 0.865. On the
test set the pattern holds: context-only neutral F1 0.693 against the forest's 0.890. So the context
alone carries much of the neutral signal, and the confound is real. Part of the forest's neutral score
probably reflects the sentence rather than the word. The forest is still 0.175 F1 higher on neutral in
dev CV, and far ahead on innate and derived, where the context gives little away.

## Cross-axis rows held out

The class is relative to the axis. *Muslim* is innate to religion, but it appears as the stereotype fill
for race/nationality items. The classifier sees no axis, so it would get the same kind of word labeled
both innate and derived. We hold out every derived or neutral row whose fill matches any kept target:
24 rows, all derived (*muslim* 15, *african* 3, *russian* 2, *muslims*, *arab*, *ethiopia*, *male* 1
each). They are saved in `data/cross_axis_probe.csv` and are not yet evaluated. Our scores therefore
exclude exactly the cases where the three-way labels are ambiguous.

## Class concentration

A few groups make up a large share of some classes. The top 10 groups hold 21.2% of derived rows
(*poor* alone has 86 of 1,271), 24.9% of innate rows (*ethiopia* has 53 of 1,291) and 12.2% of neutral
rows. Grouped cross-validation stops these words from leaking between train and test, but per-fold
scores still depend on which fold holds them. That is one source of the 0.064 standard deviation in the
forest's dev-CV macro-F1 (0.712 to 0.887 across folds). We report this and do not reweight.

## Thin class × axis cells in the test set

The test set is one group-level fold, and religion is the smallest axis. Three religion cells in the test
set have fewer than 20 rows: innate × religion has **0** rows, derived × religion 15, and neutral ×
religion 17. Every other cell has at least 48. We therefore do not report per-axis test results for
religion. The test set says nothing about innate religion words, and the overall test scores lean
on gender and race/nationality. Religion is covered only in dev CV.

## StereoSet validity

Our labels come from StereoSet's stereotype / unrelated annotations. Blodgett et al. (2021) document
problems in such benchmarks: unclear or inconsistent notions of stereotype, mislabeled or ungrammatical
items, and target groups that mix nationality, ethnicity and religion. We see some of these too: misspelled
targets (*eriteria*, *norweigan*, *columbian*, *gentlement*), a context with two blanks, and a context
that splits a word (*"BLANKed"*). Our "derived" class inherits whatever StereoSet annotators judged stereotypical.

> Blodgett, Lopez, Olteanu, Sim and Wallach. 2021. Stereotyping Norwegian Salmon: An Inventory of
> Pitfalls in Fairness Benchmark Datasets. ACL 2021. https://aclanthology.org/2021.acl-long.81

## One encoder, one layer

All vectors come from one model (`bert-base-uncased` at a pinned revision), from its final hidden
layer, mean-pooled over the span. The forest's macro-F1 (0.832 ± 0.064 dev CV, 0.891 test), against
0.603 dev CV / 0.534 test for the best shortcut baseline (part of speech), holds for this setup only.
Two conventional classifiers on the same vectors are as good or better: logistic regression reaches
0.867 ± 0.035 dev CV and 0.898 test, and k-NN 0.827 ± 0.034 / 0.882. So the signal is in the vectors,
not in the forest. Other encoders (cased, larger, RoBERTa), other layers or other pooling may place bias
differently, and we have not tested them. The forest also nearly fits its training folds (train macro-F1
0.999), so the held-out score is the one to quote.
