package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.DistanceMeasure;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface WDTW extends DistanceMeasure {

    public static final String G_FLAG = "g";

    double getG();

    void setG(double g);
}
