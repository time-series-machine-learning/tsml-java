package tsml.classifiers.distance_based.distances;

import experiments.data.DatasetLoading;
import utilities.StringUtilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

import java.util.ArrayList;
import java.util.Collections;

public class Lcss extends AbstractDistanceMeasure {

    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance

    private double epsilon = 0.01;

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    private int delta = 0;

    @Override
    public double distance(final Instance first,
                           final Instance second,
                            double limit,
                           final PerformanceStats stats) {

        checks(first, second);

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;

        // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
        if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
            // if so then reverse engineer the max LCSS distance and replace the limit
            // this is just the inverse of the return value integer rounded to an LCSS distance
            limit = (int) ((1 - limit) * aLength) + 1;
        }

        int[][] lcss = new int[aLength+1][bLength+1];

        int warpingWindow = getDelta();
        if(warpingWindow < 0) {
            warpingWindow = aLength + 1;
        }

        for(int i = 0; i < aLength; i++){
            boolean tooBig = true;
            for(int j = i-warpingWindow; j <= i+warpingWindow; j++){
                if(j < 0){
                    j = -1;
                }else if(j >= bLength){
                    j = i+warpingWindow;
                }else {
                    if(second.value(j) + this.epsilon >= first.value(i) && second.value(j) - epsilon <= first.value(i)){
                        lcss[i+1][j+1] = lcss[i][j]+1;
                    }else if(lcss[i][j+1] > lcss[i+1][j]){
                        lcss[i+1][j+1] = lcss[i][j+1];
                    }else{
                        lcss[i+1][j+1] = lcss[i+1][j];
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
        for(int j = 1; j < lcss[lcss.length-1].length; j++){
            if(lcss[lcss.length-1][j] > max){
                max = lcss[lcss.length-1][j];
            }
        }
        return 1-((double)max/aLength);
    }


    public static void main(String[] args) throws
                                           Exception {
        double[] a = {1,2,3,4,4,3,2,1};
        double[] b = {2,1,3,2,5,7,3,1};
        Lcss df = new Lcss();
        df.setDelta(5);
        df.setEpsilon(1);
        Instances data = DatasetLoading.sampleGunPoint(0)[0];
        df.distance(data.get(0), data.get(34));
//        df.distance(new DenseInstance(0, a), new DenseInstance(0, b),3);
    }

    public static final String EPSILON_FLAG = "e";
    public static final String DELTA_FLAG = "d";

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        StringUtilities.addOption(DELTA_FLAG, options, delta);
        StringUtilities.addOption(EPSILON_FLAG, options, epsilon);
        Collections.addAll(options, super.getOptions());
        return options.toArray(new String[0]);
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        super.setOptions(options);
        StringUtilities.setOption(options, EPSILON_FLAG, this::setEpsilon, Double::parseDouble);
        StringUtilities.setOption(options, DELTA_FLAG, this::setDelta, Integer::parseInt);
    }

    public int getDelta() {
        return delta;
    }

    public void setDelta(final int delta) {
        this.delta = delta;
    }
}
