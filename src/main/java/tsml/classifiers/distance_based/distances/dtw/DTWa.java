package tsml.classifiers.distance_based.distances.dtw;

import utilities.Utilities;
import weka.core.Instance;

public class DTWa extends DTWDistance {

    public DTWa(final int i) {
        super(i);
    }

    public double da(Instance ai, Instance bi, double limit) {

        double minDist;
        boolean tooBig;
        double[] a = ai.toDoubleArray();
        double[] b = bi.toDoubleArray();

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // put a or first as the longest time series
        if(bLength > aLength) {
            double[] tmp = a;
            a = b;
            b = tmp;
            aLength = a.length - 1;
            bLength = b.length - 1;
        }

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
        int windowSize;
        if(warpingWindowInPercentage) {
            if(warpingWindowPercentage < 0) {
                windowSize = aLength;
            } else {
                windowSize = (int) (warpingWindowPercentage * aLength);
            }
        } else {
            if(warpingWindow < 0) {
                windowSize = aLength;
            } else {
                windowSize = warpingWindow;
            }
        }
        windowSize++; // to include current cell

        //Extra memory than required, could limit to windowsize,
        //        but avoids having to recreate during CV
        //for varying window sizes
        double[][] distances = new double[aLength][bLength];
        setDistanceMatrix(distances);

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for(int i = 0; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = Math.min(i + windowSize + 1, bLength);
            for(int j = start; j < end; j++) {
                distances[i][j] = Double.POSITIVE_INFINITY;
            }
        }
        distances[0][0] = squaredDifference(a, 0, b, 0);
        //a is the longer series.
        //Base cases for warping 0 to all with max interval	r
        //Warp first[0] onto all second[1]...second[r+1]
        for(int j = 1; j < windowSize && j < bLength; j++) {
            distances[0][j] = distances[0][j - 1] + squaredDifference (a, 0, b, j);
        }

        //	Warp second[0] onto all first[1]...first[r+1]
        for(int i = 1; i < windowSize && i < aLength; i++) {
            distances[i][0] = distances[i - 1][0] + squaredDifference (a, i, b, 0);
        }
        //Warp the rest,
        for(int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = Math.min(i + windowSize, bLength);
            if(distances[i][start - 1] < limit) {
                tooBig = false;
            }
            for(int j = start; j < end; j++) {
                minDist = Utilities.min(distances[i][j - 1], distances[i - 1][j], distances[i - 1][j - 1]);
                distances[i][j] =
                    minDist + squaredDifference(a, i, b, j);
                if(tooBig && distances[i][j] < limit) {
                    tooBig = false;
                }
            }
            //Early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        double distance = distances[aLength - 1][bLength - 1];
        return distance;
    }

}
