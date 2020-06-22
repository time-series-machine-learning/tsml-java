package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.params.ParamSet;

public abstract class WarpingDistanceMeasure extends BaseDistanceMeasure {
    private int windowSize = -1;
    private double windowSizePercentage = -1;
    private boolean windowSizeInPercentage = false;
    // whether to keep the distance matrix
    protected boolean keepMatrix = false;

    public boolean isKeepMatrix() {
        return keepMatrix;
    }

    public void setKeepMatrix(final boolean keepMatrix) {
        this.keepMatrix = keepMatrix;
    }

    public abstract void cleanDistanceMatrix();

    @Override
    public void clean() {
        super.clean();
        cleanDistanceMatrix();
    }

    protected int findWindowSize(int aLength) {
        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int windowSize;
        if(windowSizeInPercentage) {
            if(windowSizePercentage > 1) {
                throw new IllegalArgumentException("window percentage size larger than 1");
            }
            if(windowSizePercentage < 0) {
                windowSize = aLength - 1;
            } else {
                windowSize = (int) (windowSizePercentage * (aLength - 1));
            }
        } else {
            if(this.windowSize > aLength - 1) {
                throw new IllegalArgumentException("window size larger than series length: " + this.windowSize + " vs"
                    + " " + (aLength - 1));
            }
            if(this.windowSize < 0) {
                windowSize = aLength - 1;
            } else {
                windowSize = this.windowSize;
            }
        }
        return windowSize;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(final int windowSize) {
        this.windowSize = windowSize;
        windowSizeInPercentage = false;
    }

    public double getWindowSizePercentage() {
        return windowSizePercentage;
    }

    public void setWindowSizePercentage(final double windowSizePercentage) {
        windowSizeInPercentage = true;
        this.windowSizePercentage = windowSizePercentage;
    }

    public boolean isWindowSizeInPercentage() {
        return windowSizeInPercentage;
    }

    public static final String WINDOW_SIZE_FLAG = "ws";
    public static final String WINDOW_SIZE_PERCENTAGE_FLAG = "wsp";

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_FLAG, this::setWindowSize, Integer.class);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_PERCENTAGE_FLAG, this::setWindowSizePercentage, Double.class);
    }

    @Override
    public ParamSet getParams() {
        final ParamSet paramSet = super.getParams();
        if(windowSizeInPercentage) {
            paramSet.add(WINDOW_SIZE_PERCENTAGE_FLAG, windowSizePercentage);
        } else {
            paramSet.add(WINDOW_SIZE_FLAG, windowSize);
        }
        return paramSet;
    }
}
