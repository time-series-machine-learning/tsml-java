package timeseriesweka.classifiers.distance_based.distance_measures;

import weka.core.Instance;

public class Lcss extends DistanceMeasure {

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
    public double measureDistance() {

        Instance a = getFirstInstance();
        Instance b = getSecondInstance();
        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        int[][] lcss = new int[aLength+1][bLength+1];

        int warpingWindow = getDelta();
        if(warpingWindow < 0) {
            warpingWindow = aLength + 1;
        }

        for(int i = 0; i < aLength; i++){
            for(int j = i-warpingWindow; j <= i+warpingWindow; j++){
                if(j < 0){
                    j = -1;
                }else if(j >= bLength){
                    j = i+warpingWindow;
                }else if(b.value(j) + this.epsilon >= a.value(i) && b.value(j) - epsilon <= a.value(i)){
                    lcss[i+1][j+1] = lcss[i][j]+1;
                }else if(lcss[i][j+1] > lcss[i+1][j]){
                    lcss[i+1][j+1] = lcss[i][j+1];
                }else{
                    lcss[i+1][j+1] = lcss[i+1][j];
                }

                // could maybe do an early abandon here? Not sure, investigate further
            }
        }

        int max = -1;
        for(int i = 1; i < lcss[lcss.length-1].length; i++){
            if(lcss[lcss.length-1][i] > max){
                max = lcss[lcss.length-1][i];
            }
        }
        return 1-((double)max/aLength);
    }

    public static final String EPSILON_KEY = "tolerance";
    public static final String DELTA_KEY = "delta";

    @Override
    public void setOption(final String key, final String value) {
        if(key.equals(DELTA_KEY)) {
            setDelta(Integer.parseInt(value));
        } else if(key.equals(EPSILON_KEY)) {
            setEpsilon(Double.parseDouble(value));
        }
    }

    @Override
    public String[] getOptions() {
        return new String[] {
            EPSILON_KEY,
            String.valueOf(epsilon),
            DELTA_KEY,
            String.valueOf(delta)
        };
    }

    public static final String NAME = "LCSS";

    public String toString() {
        return NAME;
    }

    public int getDelta() {
        return delta;
    }

    public void setDelta(final int delta) {
        this.delta = delta;
    }
}
