package tsml.classifiers.distance_based.distances.ddtw;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import java.util.function.Function;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.filters.Derivative;
import weka.core.DistanceFunction;
import weka.core.Instance;

public class DDTWDistance extends TransformedDistanceMeasure implements DTW {

    private final DTW dtw;

    public DDTWDistance() {
        super();
        dtw = new DTWDistance();
        setTransformer(Derivative.getGlobalCache());
        setDistanceFunction(dtw);
    }

    @Override
    public void setWarpingWindow(int warpingWindow) {
        dtw.setWarpingWindow(warpingWindow);
    }

    @Override
    public int getWarpingWindow() {
        return dtw.getWarpingWindow();
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
}
