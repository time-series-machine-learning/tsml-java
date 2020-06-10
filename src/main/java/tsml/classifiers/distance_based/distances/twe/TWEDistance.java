package tsml.classifiers.distance_based.distances.twe;

import distance.elastic.TWE;
import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.ArrayBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DoubleBasedWarpingDistanceMeasure;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 * TWED distance measure.
 * <p>
 * Contributors: goastler
 */
public class TWEDistance
    extends DoubleBasedWarpingDistanceMeasure {

    private double lambda;
    private double nu;

    public static String getNuFlag() {
        return "n";
    }

    public static String getLambdaFlag() {
        return "l";
    }

    @Override
    public double findDistance(final double[] a, final double[] b, final double limit) {
        int aLength = a.length - 1;
        int bLength = b.length - 1;

        double[] row = new double[bLength + 1];
        double[] prevRow = new double[bLength + 1];
        double[] bCosts = new double[bLength + 1];
        double min = Double.POSITIVE_INFINITY;
        // local costs initializations
        for(int j = 1; j <= bLength; j++) {
            final double bCost;
            if(j > 1) {
                bCost = Math.pow(b[j - 2] - b[j - 1], 2);
            } else {
                bCost = Math.pow(b[j - 1], 2);
            }
            bCosts[j] = bCost;
        }

        // border of the cost matrix initialization
        row[0] = 0;
        for(int j = 1; j <= bLength; j++) {
            // todo see PR334
            row[j] = row[j - 1] + bCosts[j];
            min = Math.min(min, row[j]);
        }
        if(keepMatrix) {
            matrix = new double[aLength][bLength + 1];
            System.arraycopy(row, 0, matrix[0], 0, row.length);
        }
        if(min > limit) {
            return Double.POSITIVE_INFINITY;
        }
        for(int i = 1; i <= aLength; i++) {
            {
                double[] tmp = row;
                row = prevRow;
                prevRow = tmp;
            }
            // cost init
            final double disti1;
            if(i > 1) {
                disti1 = Math.pow(a[i - 2] - a[i - 1], 2);
            } else {
                disti1 = Math.pow(a[i - 1], 2);
            }
            // border init
            double cost = prevRow[0] + disti1;
            row[0] = cost;
            min = Math.min(min, cost);
            for(int j = 1; j <= bLength; j++) {
                double htrans = Math.abs(i - j);
                if(j > 1 && i > 1) {
                    htrans += Math.abs((i - 1) - (j - 1));
                }
                double dist = Math.pow(a[i - 1] - b[j - 1], 2);
                if(i > 1 && j > 1) {
                    dist += Math.pow(a[i - 2] - b[j - 2], 2);
                }
                final double topLeft = prevRow[j - 1] + nu * htrans + dist;
                htrans = Math.min(i, 1);
                final double top = disti1 + prevRow[j] + lambda + nu * htrans;
                htrans = Math.min(j, 1);
                final double left = bCosts[j] + row[j - 1] + lambda + nu * htrans;
                cost = Math.max(topLeft, Math.max(left, top));
                row[j] = cost;
                min = Math.min(min, cost);
            }
            if(keepMatrix) {
                System.arraycopy(row, 0, matrix[i], 0, row.length);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return row[bLength];
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double getNu() {
        return nu;
    }

    public void setNu(double nu) {
        this.nu = nu;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(getNuFlag(), nu).add(getLambdaFlag(), lambda);
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, getNuFlag(), this::setNu, Double.class);
        ParamHandler.setParam(param, getLambdaFlag(), this::setLambda, Double.class);
    }

}
