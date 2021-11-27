package experiments.distance_based;

import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW_DistanceBasic;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.ERPDistance;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.LCSSDistance;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.WeightedDTW;
import weka.core.EuclideanDistance;

public class distances_debug {
    static void compareToSktime(){
        double[] x= {0,10,0,0,0,0,0,0,0,0};
        double[] y= {0,0,0,0,0,0,10,0,0,10};
        DTW_DistanceBasic dtw = new DTW_DistanceBasic();
        EuclideanDistance ed = new EuclideanDistance();
//        ERPDistance erp = new ERPDistance();
//        LCSSDistance lcss = new LCSSDistance();
//        WeightedDTW wdtw = new WeightedDTW();
        System.out.println(" DTW dist = "+dtw.distance(x,y,Double.MAX_VALUE));
    }

    public static void main(String[] args) {
        compareToSktime();
    }

}
