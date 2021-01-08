package tsml.classifiers.distance_based.utils.stats.scoring;

import weka.core.Instances;

import java.io.Serializable;
import java.util.List;

/**
 * Purpose: score the partitioning of some data into subsets.
 */
public interface SplitScorer extends Serializable {
    double findScore(Instances parent, List<Instances> children);

}
