package tsml.classifiers.distance_based.distances;

import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

public class Msm
    extends AbstractDistanceMeasure {


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
    public double distance(final Instance first,
                           final Instance second,
                           final double limit,
                           final PerformanceStats stats) {

        checkData(first, second);

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;

        double[][] cost = new double[aLength][bLength];

        // Initialization
        cost[0][0] = Math.abs(first.value(0) - second.value(0));
        for (int i = 1; i < aLength; i++) {
            cost[i][0] = cost[i - 1][0] + findCost(first.value(i), first.value(i - 1), second.value(0));
        }
        for (int i = 1; i < bLength; i++) {
            cost[0][i] = cost[0][i - 1] + findCost(second.value(i), first.value(0), second.value(i - 1));
        }

        // Main Loop
        double min;
        for (int i = 1; i < aLength; i++) {
            min = limit;
            for (int j = 1; j < bLength; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + Math.abs(first.value(i) - second.value(j));
                d2 = cost[i - 1][j] + findCost(first.value(i), first.value(i - 1), second.value(j));
                d3 = cost[i][j - 1] + findCost(second.value(j), first.value(i), second.value(j - 1));
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

                if(cost[i][j] >= limit){
                    cost[i][j] = Double.POSITIVE_INFINITY;
                }

                if(cost[i][j] < min){
                    min = cost[i][j];
                }
            }
            if(min >= limit){
                return Double.POSITIVE_INFINITY;
            }
        }
        // Output
        return cost[aLength - 1][bLength - 1];
    }

    public static final String COST_FLAG = "c";

    @Override public ParamSet getParams() {
        return super.getParams().add(COST_FLAG, cost);
    }

    @Override public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, COST_FLAG, this::setCost, Double.class);
    }

}
