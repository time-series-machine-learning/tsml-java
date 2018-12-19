// written April '16 - looks good


package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.elastic_distance_measures.TWEDistance;
//import efficient_standalone_classifiers.Eff
/**
 *
 * @author sjx07ngu
 */
public class TWE1NN extends Efficient1NN{

    
    private static final double DEGREE=2; // not bothering to set the degree in this code, it's fixed to 2 in the other anyway
    double nu=1;
    double lambda=1;

    
    protected static double[] twe_nuParams = {
        // <editor-fold defaultstate="collapsed" desc="hidden for space">
        0.00001,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,// </editor-fold>    
    };

    protected static double[] twe_lamdaParams = {
        // <editor-fold defaultstate="collapsed" desc="hidden for space">
        0,
        0.011111111,
        0.022222222,
        0.033333333,
        0.044444444,
        0.055555556,
        0.066666667,
        0.077777778,
        0.088888889,
        0.1,// </editor-fold>
    };
    
    public TWE1NN(double nu, double lambda){
        this.nu = nu;
        this.lambda = lambda;
        this.classifierIdentifier = "TWE_1NN";
        this.allowLoocv = false;
    }
    
    public TWE1NN(){
        // note: these defaults may be garbage for most measures. Should set them through CV or prior knowledge
        this.nu = 0.005;
        this.lambda = 0.5;
        this.classifierIdentifier = "TWE_1NN";
    }
    
    public final double distance(Instance first, Instance second, double cutoff){
        // note: I can't see a simple way to use the cutoff, so unfortunately there isn't one! 
        
        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if (first.classIndex() != first.numAttributes() - 1 || second.classIndex() != second.numAttributes() - 1) {
            return new TWEDistance(nu, lambda).distance(first, second, cutoff);
        }

        int m = first.numAttributes() - 1;
        int n = second.numAttributes() - 1;

        int dim = 1;
        double dist, disti1, distj1;
        double[][] ta = new double[m][dim];
        double[][] tb = new double[m][dim];
        double[] tsa = new double[m];
        double[] tsb = new double[n];
        
        // look like time staps
        
        for (int i = 0; i < tsa.length; i++) {
            tsa[i] = (i + 1);
        }
        for (int i = 0; i < tsb.length; i++) {
            tsb[i] = (i + 1);
        }

        int r = ta.length; // this is just m?!
        int c = tb.length; // so is this, but surely it should actually be n anyway
        
        
        int i, j, k;
//Copy over values
        for (i = 0; i < m; i++) {
            ta[i][0] = first.value(i);
        }
        for (i = 0; i < n; i++) {
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
        for (j = 1; j <= c; j++) {
            distj1 = 0;
            for (k = 0; k < dim; k++) {
                if (j > 1) {
//CHANGE AJB 8/1/16: Only use power of 2 for speed                      
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

        for (i = 1; i <= r; i++) {
            disti1 = 0;
            for (k = 0; k < dim; k++) {
                if (i > 1) {
                    disti1 += (ta[i - 2][k] - ta[i - 1][k]) * (ta[i - 2][k] - ta[i - 1][k]);
                } // OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                else {
                    disti1 += (ta[i - 1][k]) * (ta[i - 1][k]);
                }
            }
//OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

            Di1[i] = (disti1);

            for (j = 1; j <= c; j++) {
                dist = 0;
                for (k = 0; k < dim; k++) {
                    dist += (ta[i - 1][k] - tb[j - 1][k]) * (ta[i - 1][k] - tb[j - 1][k]);
//                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                    if (i > 1 && j > 1) {
                        dist += (ta[i - 2][k] - tb[j - 2][k]) * (ta[i - 2][k] - tb[j - 2][k]);
                    }
//                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                }
                D[i][j] = (dist);
            }
        }// for i

        // border of the cost matrix initialization
        D[0][0] = 0;
        for (i = 1; i <= r; i++) {
            D[i][0] = D[i - 1][0] + Di1[i];
        }
        for (j = 1; j <= c; j++) {
            D[0][j] = D[0][j - 1] + Dj1[j];
        }

        double dmin, htrans, dist0;
        int iback;

        for (i = 1; i <= r; i++) {
            for (j = 1; j <= c; j++) {
                htrans = Math.abs((tsa[i - 1] - tsb[j - 1]));
                if (j > 1 && i > 1) {
                    htrans += Math.abs((tsa[i - 2] - tsb[j - 2]));
                }
                dist0 = D[i - 1][j - 1] + nu * htrans + D[i][j];
                dmin = dist0;
                if (i > 1) {
                    htrans = ((tsa[i - 1] - tsa[i - 2]));
                } else {
                    htrans = tsa[i - 1];
                }
                dist = Di1[i] + D[i - 1][j] + lambda + nu * htrans;
                if (dmin > dist) {
                    dmin = dist;
                }
                if (j > 1) {
                    htrans = (tsb[j - 1] - tsb[j - 2]);
                } else {
                    htrans = tsb[j - 1];
                }
                dist = Dj1[j] + D[i][j - 1] + lambda + nu * htrans;
                if (dmin > dist) {
                    dmin = dist;
                }
                D[i][j] = dmin;
            }
        }

        dist = D[r][c];
        return dist;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void runComparison() throws Exception{
        String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";
        
//        String datasetName = "ItalyPowerDemand";
//        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
        String datasetName = "SonyAiboRobotSurface1";

        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        TWEDistance oldDtw = new TWEDistance(0.001,0.5);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);
        
        // new version
        TWE1NN dtwNew = new TWE1NN(0.001,0.5);
        dtwNew.buildClassifier(train);
        
        int correctOld = 0;
        int correctNew = 0;
        
        long start, end, oldTime, newTime;
        double pred;
               
        // classification with old MSM class and kNN
        start = System.nanoTime();
        
        correctOld = 0;
        for(int i = 0; i < test.numInstances(); i++){
            pred = knn.classifyInstance(test.instance(i));
            if(pred==test.instance(i).classValue()){
                correctOld++;
            }
        }
        end = System.nanoTime();
        oldTime = end-start;
        
        // classification with new MSM and own 1NN
        start = System.nanoTime();
        correctNew = 0;
        for(int i = 0; i < test.numInstances(); i++){
            pred = dtwNew.classifyInstance(test.instance(i));
            if(pred==test.instance(i).classValue()){
                correctNew++;
            }
        }
        end = System.nanoTime();
        newTime = end-start;
        
        System.out.println("Comparison of MSM: "+datasetName);
        System.out.println("==========================================");
        System.out.println("Old acc:    "+((double)correctOld/test.numInstances()));
        System.out.println("New acc:    "+((double)correctNew/test.numInstances()));
        System.out.println("Old timing: "+oldTime);
        System.out.println("New timing: "+newTime);
        System.out.println("Relative Performance: " + ((double)newTime/oldTime));
    }
    
  
    
      
    public static void main(String[] args) throws Exception{
        for(int i = 0; i < 10; i++){
            runComparison();
        }
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        this.nu = twe_nuParams[paramId/10];
        this.lambda = twe_lamdaParams[paramId%10];
    }

    @Override
    public String getParamInformationString() {
        return this.nu+","+this.lambda;
    }


}
