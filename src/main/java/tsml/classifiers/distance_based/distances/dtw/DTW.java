package tsml.classifiers.distance_based.distances.dtw;
/*

Purpose: interface for DTW behaviour

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;

public interface DTW extends DistanceMeasure {
    String WINDOW_FLAG = "w";

    double getWindow();
    
    void setWindow(double window);
}
