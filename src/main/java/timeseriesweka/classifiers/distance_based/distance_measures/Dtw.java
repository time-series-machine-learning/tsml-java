package timeseriesweka.classifiers.distance_based.distance_measures;


import weka.core.Instance;

public class Dtw extends DistanceMeasure {

    public int getWarpingWindow() {
        return warpingWindow;
    }

    public void setWarpingWindow(int warpingWindow) {
        this.warpingWindow = warpingWindow;
    }

    public Dtw() {

    }

    @Override
    public double measureDistance() {
        double minDist;
        boolean tooBig;

        Instance a = getFirstInstance();
        int aLength = a.numAttributes() - 1;
        Instance b = getSecondInstance();
        int bLength = b.numAttributes() - 1;
        double cutOff = getLimit();

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
        int windowSize = warpingWindow + 1; // + 1 to include the current cell
        if(warpingWindow < 0) {
            windowSize = aLength + 1;
        }
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV
//for varying window sizes
        double[][] matrixD = new double[aLength][bLength];

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for (int i = 0; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = i + windowSize + 1 < bLength ? i + windowSize + 1 :
                  bLength;
            for (int j = start; j < end; j++) {
                matrixD[i][j] = Double.MAX_VALUE;
            }
        }
        matrixD[0][0] = (a.value(0) - b.value(0)) * (a.value(0) - b.value(0));
//a is the longer series.
//Base cases for warping 0 to all with max interval	r
//Warp first[0] onto all second[1]...second[r+1]
        for (int j = 1; j < windowSize && j < bLength; j++) {
            matrixD[0][j] = matrixD[0][j - 1] + (a.value(0) - b.value(j)) * (a.value(0) - b.value(j));
        }

//	Warp second[0] onto all first[1]...first[r+1]
        for (int i = 1; i < windowSize && i < aLength; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + (a.value(i) - b.value(0)) * (a.value(i) - b.value(0));
        }
//Warp the rest,
//        System.out.println(Utilities.asString(matrixD[0]));
        for (int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = i + windowSize < bLength ? i + windowSize :
                  bLength;
            if(matrixD[i][start - 1] < cutOff) {
                tooBig = false;
            }
            for (int j = start; j < end; j++) {
                minDist = matrixD[i][j - 1];
                if (matrixD[i - 1][j] < minDist) {
                    minDist = matrixD[i - 1][j];
                }
                if (matrixD[i - 1][j - 1] < minDist) {
                    minDist = matrixD[i - 1][j - 1];
                }
                matrixD[i][j] = minDist + (a.value(i) - b.value(j)) * (a.value(i) - b.value(j));
                if (tooBig && matrixD[i][j] < cutOff) {
                    tooBig = false;
                }
            }
//            System.out.println(Utilities.asString(matrixD[i]));
            //Early abandon
            if (tooBig) {
//                System.out.println("---");
                return Double.POSITIVE_INFINITY;
            }
        }
//        System.out.println("---");
//Find the minimum distance at the end points, within the warping window.
        return matrixD[aLength - 1][bLength - 1];
    }

    private int warpingWindow;

    @Override
    public String[] getOptions() {
        return new String[] {WARPING_WINDOW_KEY, String.valueOf(warpingWindow)};
    }

    public static final String WARPING_WINDOW_KEY = "warpingWindow";

    public void setOption(String key, String value) {
        if(key.equals(WARPING_WINDOW_KEY)) {
            setWarpingWindow(Integer.parseInt(value));
        }
    }

    public static final String NAME = "DTW";

    @Override
    public String toString() {
        return NAME;
    }

}
