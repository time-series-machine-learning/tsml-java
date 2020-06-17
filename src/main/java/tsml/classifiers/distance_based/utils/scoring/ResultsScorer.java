package tsml.classifiers.distance_based.utils.scoring;

import evaluation.storage.ClassifierResults;

public interface ResultsScorer {

    double score(ClassifierResults results);
}
