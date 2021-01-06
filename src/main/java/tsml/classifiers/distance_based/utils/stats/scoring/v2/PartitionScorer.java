package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import weka.core.Instances;

import java.io.Serializable;
import java.util.List;

/**
 * Purpose: score the partitioning of some data into subsets.
 */
public interface PartitionScorer extends Serializable {
    double findScore(Instances parent, List<Instances> children);

}
