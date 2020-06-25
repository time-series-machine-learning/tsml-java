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
import tsml.transformers.Derivative;
import weka.core.Instances;

import java.util.Arrays;

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

    public static void main(String[] args) throws Exception {
        final Instances[] data = DatasetLoading.sampleGunPoint(0);
        final Instances train = data[0];

//        double[] array = train.get(0).toDoubleArray();
//        long ahcSum = 0;
//        int repeats = 100;
//        for(int i = 0; i < repeats; i++) {
//            long time = System.nanoTime();
//            int hashCode = Arrays.hashCode(array);
//            ahcSum += System.nanoTime() - time;
//        }
//        System.out.println((double) ahcSum / repeats);
//        long derSum = 0;
//        for(int i = 0; i < repeats; i++) {
//            long time = System.nanoTime();
//            double[] der = Derivative.getDerivative(array, true);
//            derSum += System.nanoTime() - time;
//        }
//        System.out.println((double) derSum / repeats);

//        DDTWDistance ddtwDistance = new DDTWDistance();
//        ddtwDistance.setInstances(train);
//        double distanceA = ddtwDistance.distance(train.get(0), train.get(1));
//        double distanceB = ddtwDistance.distance(train.get(0), train.get(1));// it should cache the transform here
//        System.out.println();

        ParamSpace space = DistanceMeasureConfigs.buildDdtwSpaceV1(train);
        System.out.println(space.toString());
        ParamSet paramSet = space.get(5);
        System.out.println(paramSet);
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
