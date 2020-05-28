package tsml.classifiers.distance_based.distances.ddtw;
/*

Purpose: derivative version of DTW.

Contributors: goastler
    
*/

import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.filters.Derivative;
import weka.core.Instances;

public class DDTWDistance extends TransformedDistanceMeasure implements DTW {

    private final DTW dtw;

    public DDTWDistance() {
        super();
        dtw = new DTWDistance();
        setTransformer(Derivative.getGlobalCache());
        setDistanceFunction(dtw);
    }

    public DDTWDistance(int warpingWindow) {
        this();
        setWarpingWindow(warpingWindow);
    }

    @Override
    public void setWarpingWindowPercentage(final double percentage) {
        dtw.setWarpingWindowPercentage(percentage);
    }

    @Override
    public double getWarpingWindowPercentage() {
        return dtw.getWarpingWindowPercentage();
    }

    @Override
    public boolean isWarpingWindowInPercentage() {
        return dtw.isWarpingWindowInPercentage();
    }

    @Override
    public int getWarpingWindow() {
        return dtw.getWarpingWindow();
    }

    @Override
    public void setWarpingWindow(int warpingWindow) {
        dtw.setWarpingWindow(warpingWindow);
    }

    @Override
    public double[][] getDistanceMatrix() {
        return dtw.getDistanceMatrix();
    }

    @Override
    public boolean isKeepDistanceMatrix() {
        return dtw.isKeepDistanceMatrix();
    }

    @Override
    public void setKeepDistanceMatrix(boolean state) {
        dtw.setKeepDistanceMatrix(state);
    }

    @Override
    public void cleanDistanceMatrix() {
        dtw.cleanDistanceMatrix();
    }

    @Override public ParamSet getParams() {
        return dtw.getParams(); // not including super params as we handle them manually in this class
    }

    @Override public void setParams(final ParamSet param) {
        dtw.setParams(param); // not including super params as we handle them manually in this class
    }
}
