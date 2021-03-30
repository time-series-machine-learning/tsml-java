package tsml.classifiers.distance_based.optimised;

import evaluation.storage.ClassifierResults;

import java.io.Serializable;

public interface ResultsScorer extends Serializable {
    
    double score(ClassifierResults results);
}
