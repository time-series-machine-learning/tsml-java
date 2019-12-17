package tsml.classifiers.distance_based.distances;


import utilities.StringUtilities;
import utilities.params.ParamDescriptor;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Dtw extends AbstractDistanceMeasure {

    public int getWarpingWindow() {
        return warpingWindow;
    }

    public void setWarpingWindow(int warpingWindow) {
        this.warpingWindow = warpingWindow;
    }

    public Dtw() {}

    public Dtw(int warpingWindow) {
        setWarpingWindow(warpingWindow);
    }

    protected double[][] distanceMatrix;
    protected boolean keepDistanceMatrix = false;
    protected int firstIntervalStart = 0;
    protected int firstIntervalLength = -1;
    protected int secondIntervalStart = 0;
    protected int secondIntervalLength = -1;

    public void setIntervalStart(int intervalStart) {
        setFirstIntervalStart(intervalStart);
        setSecondIntervalStart(intervalStart);
    }

    public void setIntervalLength(int intervalLength) {
        setFirstIntervalLength(intervalLength);
        setSecondIntervalLength(intervalLength);
    }

    public int getFirstIntervalStart() {
        return firstIntervalStart;
    }

    public void setFirstIntervalStart(final int firstIntervalStart) {
        this.firstIntervalStart = firstIntervalStart;
    }

    public int getFirstIntervalLength() {
        return firstIntervalLength;
    }

    public void setFirstIntervalLength(final int firstIntervalLength) {
        this.firstIntervalLength = firstIntervalLength;
    }

    public int getSecondIntervalStart() {
        return secondIntervalStart;
    }

    public void setSecondIntervalStart(final int secondIntervalStart) {
        this.secondIntervalStart = secondIntervalStart;
    }

    public int getSecondIntervalLength() {
        return secondIntervalLength;
    }

    public void setSecondIntervalLength(final int secondIntervalLength) {
        this.secondIntervalLength = secondIntervalLength;
    }

    public double[][] getDistanceMatrix() {
        return distanceMatrix;
    }

    public void setDistanceMatrix(final double[][] distanceMatrix) {
        this.distanceMatrix = distanceMatrix;
    }

    public boolean isKeepDistanceMatrix() {
        return keepDistanceMatrix;
    }

    public void setKeepDistanceMatrix(final boolean keepDistanceMatrix) {
        this.keepDistanceMatrix = keepDistanceMatrix;
    }

    protected void finaliseDistanceMatrix() {
        if(!keepDistanceMatrix) {
            distanceMatrix = null;
        }
    }

    @Override
    public double distance(final Instance first,
                                  final Instance second,
                                  final double limit,
                                  final PerformanceStats stats) {
        checks(first, second);


        double minDist;
        boolean tooBig;

        int aLength = firstIntervalLength;
        if(aLength < 0) {
            aLength = first.numAttributes() - 1;
        }
        int bLength = secondIntervalLength;
        if(bLength < 0) {
            bLength = second.numAttributes() - 1;
        }

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
        int windowSize = warpingWindow + 1; // + 1 to include the current cell
        if(warpingWindow < 0) {
            windowSize = aLength + 1; // todo how would this work for unequal length time series?
        }
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV
//for varying window sizes
        distanceMatrix = new double[aLength][bLength];

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for (int i = firstIntervalStart; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = Math.min(i + windowSize + 1, bLength);
            for (int j = start; j < end; j++) {
                distanceMatrix[i][j] = Double.POSITIVE_INFINITY;
            }
        }
        distanceMatrix[0][0] =
            (first.value(firstIntervalStart) - second.value(secondIntervalStart)) * (first.value(firstIntervalStart) - second.value(secondIntervalStart));
//a is the longer series.
//Base cases for warping 0 to all with max interval	r
//Warp first[0] onto all second[1]...second[r+1]
        for (int j = 1; j < windowSize && j < bLength; j++) {
            distanceMatrix[0][j] =
                distanceMatrix[0][j - 1] + (first.value(firstIntervalStart) - second.value(j + secondIntervalStart)) * (first.value(firstIntervalStart) - second.value(j + secondIntervalStart));
        }

//	Warp second[0] onto all first[1]...first[r+1]
        for (int i = 1; i < windowSize && i < aLength; i++) {
            distanceMatrix[i][0] =
                distanceMatrix[i - 1][0] + (first.value(i + firstIntervalStart) - second.value(secondIntervalStart)) * (first.value(i + firstIntervalStart) - second.value(secondIntervalStart));
        }
//Warp the rest,
        for (int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = Math.min(i + windowSize, bLength);
            if(distanceMatrix[i][start - 1] < limit) {
                tooBig = false;
            }
            for (int j = start; j < end; j++) {
                minDist = distanceMatrix[i][j - 1];
                if (distanceMatrix[i - 1][j] < minDist) {
                    minDist = distanceMatrix[i - 1][j];
                }
                if (distanceMatrix[i - 1][j - 1] < minDist) {
                    minDist = distanceMatrix[i - 1][j - 1];
                }
                distanceMatrix[i][j] =
                    minDist + (first.value(i + firstIntervalStart) - second.value(j + secondIntervalStart)) * (first.value(i + firstIntervalStart) - second.value(j + secondIntervalStart));
                if (tooBig && distanceMatrix[i][j] < limit) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                finaliseDistanceMatrix();
                return Double.POSITIVE_INFINITY;
            }
        }
//Find the minimum distance at the end points, within the warping window.
        double distance = distanceMatrix[aLength - 1][bLength - 1];
        finaliseDistanceMatrix();
        return distance;
    }

    protected int warpingWindow;
    protected final ParamDescriptor<Integer> warpingWindowParam = new ParamDescriptor<>("w", this::getWarpingWindow,
                                                                                        this::setWarpingWindow,
                                                                                        Integer::parseInt);

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        super.setOptions(options);
        StringUtilities.setOption(options, WARPING_WINDOW_FLAG, this::setWarpingWindow, Integer::parseInt);
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        StringUtilities.addOption(WARPING_WINDOW_FLAG, options, warpingWindow);
        Collections.addAll(options, super.getOptions());
        return options.toArray(new String[0]);
    }

    public static final String WARPING_WINDOW_FLAG = "w";

    @Override public ParamSet getParams() {
        ParamSet paramSet = new ParamSet();
        paramSet.add(WARPING_WINDOW_FLAG, warpingWindow);
        return paramSet;
    }

    @Override public void setParams(final ParamSet params) {
        ParamHandler.setParam(params, WARPING_WINDOW_FLAG, this::setWarpingWindow, Integer.class);
    }
}
