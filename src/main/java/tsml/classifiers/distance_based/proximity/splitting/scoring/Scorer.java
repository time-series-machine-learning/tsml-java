package tsml.classifiers.distance_based.proximity.splitting.scoring;

import utilities.Utilities;
import weka.core.Instances;

import java.util.List;

/**
 * Purpose: score the partitioning of some data into subsets.
 */
public interface Scorer {
    double findScore(Instances parent, List<Instances> parts);
}
