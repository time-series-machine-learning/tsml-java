package tsml.classifiers.distance_based.distances.ddtw;
/*

Purpose: derivative version of DTW.

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.transformers.Derivative;
import weka.core.Instances;

public class DDTWDistance extends TransformedDistanceMeasure implements DTW {

    private final DTW dtw;

    public DDTWDistance() {
        super();
        dtw = new DTWDistance();
        setTransformer(Derivative.getGlobalCache());
        setDistanceFunction(dtw);
    }

    public DDTWDistance(int WindowSize) {
        this();
        setWindowSize(WindowSize);
    }

    @Override
    public void setWindowSizePercentage(final double percentage) {
        dtw.setWindowSizePercentage(percentage);
    }

    @Override
    public double getWindowSizePercentage() {
        return dtw.getWindowSizePercentage();
    }

    @Override
    public boolean isWindowSizeInPercentage() {
        return dtw.isWindowSizeInPercentage();
    }

    @Override
    public int getWindowSize() {
        return dtw.getWindowSize();
    }

    @Override
    public void setWindowSize(int WindowSize) {
        dtw.setWindowSize(WindowSize);
    }

    @Override
    public double[][] getMatrix() {
        return dtw.getMatrix();
    }

    @Override
    public boolean isKeepMatrix() {
        return dtw.isKeepMatrix();
    }

    @Override
    public void setKeepMatrix(boolean state) {
        dtw.setKeepMatrix(state);
    }

    @Override
    public void cleanDistanceMatrix() {
        dtw.cleanDistanceMatrix();
    }
}
