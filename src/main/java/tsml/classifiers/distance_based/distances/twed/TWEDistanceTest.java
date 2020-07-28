package tsml.classifiers.distance_based.distances.twed;

import java.util.Random;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest.DistanceTester;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.GridSearchIterator;
import weka.core.Instance;
import weka.core.Instances;

public class TWEDistanceTest {


    private static DistanceTester buildDistanceFinder() {
        return new DistanceTester() {
            private ParamSpace space;
            private Instances data;

            @Override
            public void findDistance(final Random random, final Instances data, final Instance ai,
                final Instance bi, final double limit) {
                if(data != this.data) {
                    this.data = data;
                    space = TWEDistanceConfigs.buildTWEDParams();
                }
                final GridSearchIterator iterator = new GridSearchIterator(space);
//                                                int i = 0;
                while(iterator.hasNext()) {
//                                                            System.out.println("i:" + i++);
                    final ParamSet paramSet = iterator.next();
                    final double nu = (double) paramSet.get(TWEDistance.NU_FLAG).get(0);
                    final double lambda = (double) paramSet.get(TWEDistance.LAMBDA_FLAG).get(0);
                    // doesn't test window, MSM originally doesn't have window
                    //                    final int window = (int) paramSet.get(MSMDistance.).get(0);
                    final TWEDistance df = new TWEDistance();
                    df.setLambda(lambda);
                    df.setNu(nu);
                    Assert.assertEquals(df.distance(ai, bi, limit), origTwed(ai, bi, limit, lambda, nu), 0);
                }
            }
        };
    }

    @Test
    public void testRandomDataset() {
        ERPDistanceTest.testDistanceFunctionsOnRandomDataset(buildDistanceFinder());
    }

    @Test
    public void testGunPoint() throws Exception {
        ERPDistanceTest.testDistanceFunctionsOnGunPoint(buildDistanceFinder());
    }

    @Test
    public void testItalyPowerDemand() throws Exception {
        ERPDistanceTest.testDistanceFunctionsOnItalyPowerDemand(buildDistanceFinder());
    }

    @Test
    public void testBeef() throws Exception {
        ERPDistanceTest.testDistanceFunctionsOnBeef(buildDistanceFinder());
    }

    private static double origTwed(Instance a, Instance b, double limit, double lambda, double nu) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;
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
            ta[i][0] = a.value(i);
        }
        for(i = 0; i < bLength; i++) {
            tb[i][0] = b.value(i);
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
            //            D[i][0] = Double.POSITIVE_INFINITY;
            D[i][0] = D[i - 1][0] + Di1[i];
        }
        for(j = 1; j <= c; j++) {
            //            D[0][j] = Double.POSITIVE_INFINITY;
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
            double min = Double.POSITIVE_INFINITY;
            for(int m = 0; m < D[i].length; m++) {
                min = Math.min(min, D[i][m]);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        dist = D[r][c];
        return dist;
    }

}
