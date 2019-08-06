package timeseriesweka.classifiers.distance_based.distances.lcss;

import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;

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
    public double distance() {

        double[] first = getTarget();
        double[] second = getCandidate();
        // todo cleanup
        // todo trim memory to window by window
        // todo early abandon
        int m = first.length;
        int n = second.length;

        int[][] lcss = new int[m+1][n+1];

        int warpingWindow = getDelta();
        if(warpingWindow < 0) {
            warpingWindow = first.length + 1;
        }

        for(int i = 0; i < m; i++){
            for(int j = i-warpingWindow; j <= i+warpingWindow; j++){
                if(j < 0){
                    j = -1;
                }else if(j >= n){
                    j = i+warpingWindow;
                }else if(second[j]+this.epsilon >= first[i] && second[j] - epsilon <= first[i]){
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
        return 1-((double)max/m);
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
