package tsml.classifiers.distance_based.distances.dtw;
/*

Purpose: interface for DTW behaviour

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasureable;

public interface DTW extends DistanceMeasureable {
    void setWarpingWindow(int warpingWindow);
    int getWarpingWindow();
    void setWarpingWindowPercentage(double percentage);
    double getWarpingWindowPercentage();
    boolean isWarpingWindowInPercentage();
    static String getWarpingWindowFlag() {
        return "w";
    }
    double[][] getDistanceMatrix();
    boolean isKeepDistanceMatrix();
    void setKeepDistanceMatrix(boolean state);
    void cleanDistanceMatrix();
}
