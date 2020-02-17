package tsml.classifiers.distance_based.pf;

import weka.core.Instances;

import java.util.List;

public interface Split {
    List<Instances> getSplitOutputData();
    Instances getSplitInputData();
    default void cleanUpSplit() {

    }
    double getSplitScore();
    void setSplitScore(double score);
}
