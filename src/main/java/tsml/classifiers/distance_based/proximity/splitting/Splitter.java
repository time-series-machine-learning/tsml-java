package tsml.classifiers.distance_based.proximity.splitting;

import weka.core.Instances;

/**
 * Purpose: a splitter takes a given set of data and splits it into one to many partitions.
 * <p>
 * Contributors: goastler
 */
public interface Splitter {

    public Split buildSplit(Instances data);
}
