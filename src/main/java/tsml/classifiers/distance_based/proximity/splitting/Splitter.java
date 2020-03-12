package tsml.classifiers.distance_based.proximity.splitting;

import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class Splitter {

    public abstract Split buildSplit(Instances data);
}
