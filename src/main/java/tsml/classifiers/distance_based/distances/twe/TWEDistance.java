package tsml.classifiers.distance_based.distances.twe;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

/**
 * TWED distance measure.
 * <p>
 * Contributors: goastler
 */
public class TWEDistance
    extends BaseDistanceMeasure {

    private double lambda;
    private double nu;

    public static String getNuFlag() {
        return "n";
    }

    public static String getLambdaFlag() {
        return "l";
    }

    @Override
    public double distance(final Instance first,
        final Instance second,
        final double limit,
        final PerformanceStats stats) {

        checkData(first, second);

        /*This code is faithful to the c version, so uses a redundant
 * Multidimensional representation. The c code does not describe what the
            arguments
 * tsB and tsA are. We assume they are the time stamps (i.e. index sets),
 * and initialise them accordingly.
 */

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;
        int dim = 1;
        double dist, disti1, distj1;
        double[][] ta = new double[aLength][dim];
        double[][] tb = new double[bLength][dim];
        double[] tsa = new double[aLength];
        double[] tsb = new double[bLength];
        for(int i = 0; i < tsa.length; i++) {
            tsa[i] = (i + 1);
        }
        for(int i = 0; i < tsb.length; i++) {
            tsb[i] = (i + 1);
        }

        int r = ta.length;
        int c = tb.length;
        int i, j, k;
        //Copy over values
        for(i = 0; i < aLength; i++) {
            ta[i][0] = first.value(i);
        }
        for(i = 0; i < bLength; i++) {
            tb[i][0] = second.value(i);
        }

        /* allocations in c
	double **D = (double **)calloc(r+1, sizeof(double*));
	double *Di1 = (double *)calloc(r+1, sizeof(double));
	double *Dj1 = (double *)calloc(c+1, sizeof(double));
	for(i=0; i<=r; i++) {
		D[i]=(double *)calloc(c+1, sizeof(double));
	}
*/
        double[][] D = new double[r + 1][c + 1];
        double[] Di1 = new double[r + 1];
        double[] Dj1 = new double[c + 1];
        // local costs initializations
        for(j = 1; j <= c; j++) {
            distj1 = 0;
            for(k = 0; k < dim; k++) {
                if(j > 1) {
                    //CHANGE AJB 8/1/16: Only use power of 2 for speed up,
                    distj1 += (tb[j - 2][k] - tb[j - 1][k]) * (tb[j - 2][k] - tb[j - 1][k]);
                    // OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
                    // in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
                } else {
                    distj1 += tb[j - 1][k] * tb[j - 1][k];
                }
            }
            //OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
            Dj1[j] = (distj1);
        }

        for(i = 1; i <= r; i++) {
            disti1 = 0;
            for(k = 0; k < dim; k++) {
                if(i > 1) {
                    disti1 += (ta[i - 2][k] - ta[i - 1][k]) * (ta[i - 2][k] - ta[i - 1][k]);
                }
                // OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                else {
                    disti1 += (ta[i - 1][k]) * (ta[i - 1][k]);
                }
            }
            //OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

            Di1[i] = (disti1);

            for(j = 1; j <= c; j++) {
                dist = 0;
                for(k = 0; k < dim; k++) {
                    dist += (ta[i - 1][k] - tb[j - 1][k]) * (ta[i - 1][k] - tb[j - 1][k]);
                    //                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                    if(i > 1 && j > 1) {
                        dist += (ta[i - 2][k] - tb[j - 2][k]) * (ta[i - 2][k] - tb[j - 2][k]);
                    }
                    //                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                }
                D[i][j] = (dist);
            }
        }// for i

        // border of the cost matrix initialization
        D[0][0] = 0;
        for(i = 1; i <= r; i++) {
            D[i][0] = D[i - 1][0] + Di1[i];
        }
        for(j = 1; j <= c; j++) {
            D[0][j] = D[0][j - 1] + Dj1[j];
        }

        double dmin, htrans, dist0;
        int iback;

        for(i = 1; i <= r; i++) {
            for(j = 1; j <= c; j++) {
                htrans = Math.abs((tsa[i - 1] - tsb[j - 1]));
                if(j > 1 && i > 1) {
                    htrans += Math.abs((tsa[i - 2] - tsb[j - 2]));
                }
                dist0 = D[i - 1][j - 1] + nu * htrans + D[i][j];
                dmin = dist0;
                if(i > 1) {
                    htrans = ((tsa[i - 1] - tsa[i - 2]));
                } else {
                    htrans = tsa[i - 1];
                }
                dist = Di1[i] + D[i - 1][j] + lambda + nu * htrans;
                if(dmin > dist) {
                    dmin = dist;
                }
                if(j > 1) {
                    htrans = (tsb[j - 1] - tsb[j - 2]);
                } else {
                    htrans = tsb[j - 1];
                }
                dist = Dj1[j] + D[i][j - 1] + lambda + nu * htrans;
                if(dmin > dist) {
                    dmin = dist;
                }
                D[i][j] = dmin;
            }
        }

        dist = D[r][c];
        return dist;
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
