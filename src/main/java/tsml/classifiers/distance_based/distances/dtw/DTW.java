package tsml.classifiers.distance_based.distances.dtw;
/*

Purpose: interface for DTW behaviour

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasure;

public interface DTW extends DistanceMeasure {
    void setWindowSize(int windowSize);
    int getWindowSize();
    void setWindowSizePercentage(double percentage);
    double getWindowSizePercentage();
    boolean isWindowSizeInPercentage();
    double[][] getMatrix();
    boolean isKeepMatrix();
    void setKeepMatrix(boolean state);
    void cleanDistanceMatrix();
}
