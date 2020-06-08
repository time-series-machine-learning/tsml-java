package tsml.classifiers.distance_based.distances;

public abstract class WarpingDistanceMeasure extends ArrayBasedDistanceMeasure {
    private int windowSize = -1;
    private double windowSizePercentage = -1;
    private boolean windowSizeInPercentage = false;
    // the distance matrix produced by the distance function
    protected double[][] matrix;
    // whether to keep the distance matrix
    protected boolean keepMatrix = false;

    public double[][] getMatrix() {
        return matrix;
    }

    protected void setMatrix(double[][] matrix) {
        if(keepMatrix) {
            this.matrix = matrix;
        }
    }

    public boolean isKeepMatrix() {
        return keepMatrix;
    }

    public void setKeepMatrix(final boolean keepMatrix) {
        this.keepMatrix = keepMatrix;
    }

    public void cleanDistanceMatrix() {
        matrix = null;
    }

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

    @Override
    public abstract double distance(double[] a, double[] b, final double limit);

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

}
