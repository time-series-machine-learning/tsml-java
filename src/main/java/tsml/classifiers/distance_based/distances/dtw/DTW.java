package tsml.classifiers.distance_based.distances.dtw;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasureable;

public interface DTW extends DistanceMeasureable {
    void setWarpingWindow(int warpingWindow);
    int getWarpingWindow();
    static String getWarpingWindowFlag() {
        return "w";
    }
    double[][] getDistanceMatrix();
    boolean isKeepDistanceMatrix();
    void setKeepDistanceMatrix(boolean state);
    void cleanDistanceMatrix();
}
