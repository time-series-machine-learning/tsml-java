package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import evaluation.storage.ClassifierResults;

public interface ResultsScorer {

    double score(ClassifierResults results);
}
