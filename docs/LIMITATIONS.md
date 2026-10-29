# Limitations (draft)

Numbers are from the pinned StereoSet and BERT revisions, with 5-fold GroupKFold on 3,857 rows (CHANGE-02
check-in). They are means over folds.

## Neutral context confound

Every neutral row comes from StereoSet's *unrelated* sentence, which is often nonsensical (*"Ethiopian
men are floating"*). A model could learn to spot the odd sentence instead of the word. To test this we
trained the same forest on a context vector that leaves the word out. That context-only model reaches a
neutral F1 of 0.721 (precision 0.709, recall 0.734). It is near chance on innate (0.402) and derived
(0.282), and its macro-F1 is 0.468 ± 0.019. The full forest reaches a neutral F1 of 0.879. So the context
alone carries much of the neutral signal, and the confound is real. Part of the forest's neutral score
probably reflects the sentence rather than the word. The forest is still 0.158 F1 higher on neutral, and
far ahead on innate and derived, where the context gives little away.

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
scores still depend on which fold holds them. That is one source of the 0.034 standard deviation in the
forest's macro-F1 (0.822 to 0.907 across folds). We report this and do not reweight.

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
layer, mean-pooled over the span. The forest's held-out macro-F1 of 0.861 ± 0.034, against 0.603 for the
best baseline (single direction), holds for this setup only. Other encoders (cased, larger, RoBERTa),
other layers or other pooling may place bias differently, and we have not tested them. The forest also
fits its training folds perfectly (train macro-F1 1.000), so the held-out score is the one to quote.
