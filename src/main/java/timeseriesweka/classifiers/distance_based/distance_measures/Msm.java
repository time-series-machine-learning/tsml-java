package timeseriesweka.classifiers.distance_based.distance_measures;

import weka.core.Instance;

public class Msm
    extends DistanceMeasure {


    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }

    private double cost = 1;

    private double findCost(double new_point, double x, double y) {
        double dist = 0;

        if (((x <= new_point) && (new_point <= y)) ||
            ((y <= new_point) && (new_point <= x))) {
            dist = getCost();
        } else {
            dist = getCost() + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
        }

        return dist;
    }

    @Override
    public double measureDistance() {
        Instance a = getFirstInstance();
        int aLength = a.numAttributes() - 1;
        Instance b = getSecondInstance();
        int bLength = b.numAttributes() - 1;

        double[][] cost = new double[aLength][bLength];

        // Initialization
        cost[0][0] = Math.abs(a.value(0) - b.value(0));
        for (int i = 1; i < aLength; i++) {
            cost[i][0] = cost[i - 1][0] + findCost(a.value(i), a.value(i - 1), b.value(0));
        }
        for (int i = 1; i < bLength; i++) {
            cost[0][i] = cost[0][i - 1] + findCost(b.value(i), a.value(0), b.value(i-1));
        }

        // Main Loop
        double min;
        double cutOffValue = getLimit();
        for (int i = 1; i < aLength; i++) {
            min = cutOffValue;
            for (int j = 1; j < bLength; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + Math.abs(a.value(i) - b.value(j));
                d2 = cost[i - 1][j] + findCost(a.value(i), a.value(i-1), b.value(j));
                d3 = cost[i][j - 1] + findCost(b.value(j), a.value(i), b.value(j-1));
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

                if(cost[i][j] >=cutOffValue){
                    cost[i][j] = Double.POSITIVE_INFINITY;
                }

                if(cost[i][j] < min){
                    min = cost[i][j];
                }
            }
            if(min >= cutOffValue){
                return Double.POSITIVE_INFINITY;
            }
        }
        // Output
        return cost[aLength - 1][bLength - 1];
    }

    public static final String COST_KEY = "cost";

    @Override
    public void setOption(final String key, final String value) {
        if (key.equals(COST_KEY)) {
            setCost(Double.parseDouble(value));
        }
    }

    @Override
    public String[] getOptions() {
        return new String[] {
            COST_KEY,
            String.valueOf(cost)
        };
    }


    public static final String NAME = "MSM";

    @Override
    public String toString() {
        return NAME;
    }
}
