package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.DistanceMeasureable;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface WDTW extends DistanceMeasureable {

    static String getGFlag() {
        return "g";
    }

    double getG();

    void setG(double g);
}
