package tsml.classifiers.distance_based.proximity.tmp;

import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class Splitter {

    abstract Split split(Instances data);
}
