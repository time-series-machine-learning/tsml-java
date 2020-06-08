package tsml.classifiers.distance_based.distances.lcss;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

/**
 * LCSS distance measure.
 * <p>
 * Contributors: goastler
 */
public class LCSSDistance extends BaseDistanceMeasure {

    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance

    private double epsilon = 0.01;
    private int delta = 0;

    public static String getEpsilonFlag() {
        return "e";
    }

    public static String getDeltaFlag() {
        return "d";
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public double distance(double[] first, double[] second){

        int m = first.length;
        int n = second.length;

        int[][] lcss = new int[m+1][n+1];
        int[][] lastX = new int[m+1][n+1];
        int[][] lastY = new int[m+1][n+1];


        for(int i = 0; i < m; i++){
            for(int j = i-delta; j <= i+delta; j++){
                //                System.out.println("here");
                if(j < 0 || j >= n){
                    //do nothing
                }else if(second[j]+this.epsilon >= first[i] && second[j]-epsilon <= first[i]){
                    lcss[i+1][j+1] = lcss[i][j]+1;
                    lastX[i+1][j+1] = i;
                    lastY[i+1][j+1] = j;
                }else if(lcss[i][j+1] > lcss[i+1][j]){
                    lcss[i+1][j+1] = lcss[i][j+1];
                    lastX[i+1][j+1] = i;
                    lastY[i+1][j+1] = j+1;
                }else{
                    lcss[i+1][j+1] = lcss[i+1][j];
                    lastX[i+1][j+1] = i+1;
                    lastY[i+1][j+1] = j;
                }
            }
        }

        int max = -1;
        for(int i = 1; i < lcss[lcss.length-1].length; i++){
            if(lcss[lcss.length-1][i] > max){
                max = lcss[lcss.length-1][i];
            }
        }
        return 1-((double)max/m);
    }

    @Override
    public double distance(final Instance a,
        final Instance b,
        double limit,
        final PerformanceStats stats) {

        checkData(a, b);

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (int) ((1 - limit) * aLength) + 1;
        }

        int[][] lcss = new int[aLength + 1][bLength + 1];

        int warpingWindow = getDelta();
        if(warpingWindow < 0) {
            warpingWindow = aLength + 1;
        }

        for(int i = 0; i < aLength; i++) {
            boolean tooBig = true;
            for(int j = i - warpingWindow; j <= i + warpingWindow; j++) {
                if(j < 0) {
                    j = -1;
                } else if(j >= bLength) {
                    j = i + warpingWindow;
                } else {
                    if(b.value(j) + this.epsilon >= a.value(i) && b.value(j) - epsilon <= a
                        .value(i)) {
                        lcss[i + 1][j + 1] = lcss[i][j] + 1;
//                    } else if(lcss[i][j + 1] > lcss[i + 1][j]) {
//                        lcss[i + 1][j + 1] = lcss[i][j + 1];
//                    } else {
//                        lcss[i + 1][j + 1] = lcss[i + 1][j];
                    }
                    else {
                        lcss[i + 1][j + 1] = Utilities.max(lcss[i + 1][j], lcss[i][j], lcss[i][j + 1]);
                    }
                    // if this value is less than the limit then fast-fail the limit overflow
                    if(tooBig && lcss[i + 1][j + 1] < limit) {
                        tooBig = false;
                    }
                }
            }

            // if no element is lower than the limit then early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }

        }
        //        System.out.println(ArrayUtilities.toString(lcss, ",", System.lineSeparator()));

        int max = -1;
        for(int j = 1; j < lcss[lcss.length - 1].length; j++) {
            if(lcss[lcss.length - 1][j] > max) {
                max = lcss[lcss.length - 1][j];
            }
        }
        return 1 - ((double) max / aLength);
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(getEpsilonFlag(), epsilon).add(getDeltaFlag(), delta);
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, getEpsilonFlag(), this::setEpsilon, Double.class);
        ParamHandler.setParam(param, getDeltaFlag(), this::setDelta, Integer.class);
    }

    public int getDelta() {
        return delta;
    }

    public void setDelta(final int delta) {
        this.delta = delta;
    }


    public double lcssDist(double[] a, double[] b, double epsilon) {
        final int m = a.length + 1;
        final int n = b.length + 1;
        final int[][] matrix = new int[m][n];

        final int[] col1 = matrix[0];
        if(approxEqual(a[0], b[0], epsilon)) {
            col1[0] = 1;
        } else {
            col1[0] = 0;
        }

        for(int i = 1; i < a.length; i++) {
            final int[] row = matrix[i];
            if(approxEqual(a[i], b[0], epsilon)) {
                row[0] = 1;
            } else {
                row[0] = matrix[i - 1][0];
            }
        }

        for(int j = 1; j < b.length; j++) {
            if(approxEqual(a[0], b[j], epsilon)) {
                col1[j] = 1;
            } else {
                col1[j] = col1[j - 1];
            }
        }

        for(int i = 1; i < m; i++) {
            final int[] row = matrix[i];
            final int[] prevRow = matrix[i - 1];
            for(int j = 1; j < n; j++) {
                if(approxEqual(a[i - 1], b[j - 1], epsilon)) {
                    row[j] = prevRow[j - 1] + 1;
                } else {
                    row[j] = Utilities.max(row[j - 1], prevRow[j], prevRow[j - 1]);
                }
            }
        }

        return 1 - (double) matrix[a.length - 1][b.length - 1] / Utilities.min(a.length, b.length);
    }

    public static boolean approxEqual(double a, double b, double epsilon) {
        return Math.abs(a - b) <= epsilon;
    }
}
