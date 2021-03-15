package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTW;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface WDTW extends DistanceMeasure {

    String G_FLAG = "g";

    double getG();

    void setG(double g);
}
