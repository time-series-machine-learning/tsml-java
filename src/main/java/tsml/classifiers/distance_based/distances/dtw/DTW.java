package tsml.classifiers.distance_based.distances.dtw;
/*

Purpose: interface for DTW behaviour

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;

public interface DTW extends ParamHandler {
    String WINDOW_SIZE_FLAG = "w";

    double getWindowSize();
    
    void setWindowSize(double windowSize);
}
