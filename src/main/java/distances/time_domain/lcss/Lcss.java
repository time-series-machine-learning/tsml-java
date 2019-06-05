package distances.time_domain.lcss;

import distances.time_domain.dtw.Dtw;
import evaluation.tuning.ParameterSpace;
import utilities.ArrayUtilities;
import utilities.StatisticalUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

public class Lcss extends Dtw {

    public static final double DEFAULT_TOLERANCE = 0.01;

    public Lcss(double tolerance, int warpingWindow) {
        super(warpingWindow);
        setTolerance(tolerance);
    }

    // delta === warp
    // epsilon === diff between two values before they're considered the same AKA tolerance

    public Lcss() {
        super();
        setTolerance(DEFAULT_TOLERANCE);
    }

    private double tolerance;

    public double getTolerance() {
        return tolerance;
    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    @Override
    public double distance(Instance a,
                           Instance b,
                           final double cutOff) {

        double[] first = Utilities.extractTimeSeries(a);
        double[] second = Utilities.extractTimeSeries(b);
        // todo cleanup
        // todo trim memory to window by window
        // todo early abandon
        int m = first.length;
        int n = second.length;

        int[][] lcss = new int[m+1][n+1];

        int warpingWindow = (int) (this.getWarpingWindow() * first.length);

        for(int i = 0; i < m; i++){
            for(int j = i-warpingWindow; j <= i+warpingWindow; j++){
                if(j < 0){
                    j = -1;
                }else if(j >= n){
                    j = i+warpingWindow;
                }else if(second[j]+this.tolerance >= first[i] && second[j]-tolerance <=first[i]){
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

    public static final String TOLERANCE_KEY = "tolerance";

    @Override
    public void setOptions(String[] options) {
        super.setOptions(options);
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if(key.equals(TOLERANCE_KEY)) {
                setTolerance(Double.parseDouble(value));
            }
        }
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            TOLERANCE_KEY,
            String.valueOf(tolerance)
        });
    }

    public static final String NAME = "LCSS";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace discreteParameterSpace(Instances instances) {
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std*0.2;
        double[] toleranceValues = ArrayUtilities.incrementalRange(stdFloor, std, 10);
        int[] warpingWindowValues = ArrayUtilities.incrementalRange(0, (instances.numAttributes() - 1) / 4, 10);
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        parameterSpace.addParameter(WARPING_WINDOW_KEY, warpingWindowValues);
        parameterSpace.addParameter(TOLERANCE_KEY, toleranceValues);
        return parameterSpace;
    }


}
