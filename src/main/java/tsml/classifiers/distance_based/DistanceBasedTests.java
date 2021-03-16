package tsml.classifiers.distance_based;

import distance.elastic.Euclidean;
import distance.elastic.TWE;
import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.legacy.elastic_ensemble.LCSS1NN;
import weka.classifiers.Classifier;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Tests to mimic those done is sktime unit tests. This is not a unit test for tsml, but could become so
 *
 */
public class DistanceBasedTests {
    public static void main(String[] args) throws Exception {
        kNNTest();
    }

    private static int countCorrect(Classifier c, Instances train) throws Exception {
        int correct=0;
        for(Instance ins: train){
            double x = c.classifyInstance(ins);
            if(x == ins.classValue())
                correct++;
        }
        return correct;
    }
    public static void kNNTest() throws Exception {
        String dataDir=DatasetLoading.BAKED_IN_TSC_DATA_PATH;
        String problem="ArrowHead";
//Count the number correct for ArrowHead 1-NN for the following distance functions with default parameters
        Instances train = DatasetLoading.loadData(dataDir+problem+"/"+problem+"_TRAIN");
        Instances test = DatasetLoading.loadData(dataDir+problem+"/"+problem+"_TEST");
        int numDistances=8;
        int[] correct = new int[numDistances];
        DistanceFunction[] measures = new DistanceFunction[numDistances];
        measures[0]=new EuclideanDistance();
        measures[1]=new DTWDistance();
        measures[2]=new MSMDistance();
        measures[3]=new WDTWDistance();
        measures[4]=new ERPDistance();
        measures[5]=new LCSSDistance();
        measures[6]=new TWEDistance();
        measures[7]= WDTWDistanceConfigs.newWDDTWDistance();

        ((LCSSDistance)measures[5]).setWindowSize(3);
        ((LCSSDistance)measures[5]).setEpsilon(0.05);

        int i=0;
        for(DistanceFunction d:measures) {
            KNN knn = new KNN();
            knn.setSeed(0);
            knn.setDistanceFunction(d);
            knn.buildClassifier(train);
            correct[i] = countCorrect(knn, test);
            System.out.println("Distance measure  " + d + " gets " + correct[i++] + " correct out of "+test.numInstances());
        }

/*
        LCSS1NN jayLCSS = new LCSS1NN(3,0.05);
        KNN knn = new KNN();
        knn.setSeed(0);
        knn.setDistanceFunction(measures[5]);
        jayLCSS.buildClassifier(train);
        knn.buildClassifier(train);
        Instance wrongUn=test.instance(2);
        int c=16;
        System.out.println(" WRONG UN INDEX = "+c);
//        int county = countCorrect(jayLCSS, test);
//        System.out.println("Jay LCSS gets " + county + " correct out of "+test.numInstances());
        double min1=Double.MAX_VALUE,min2=Double.MAX_VALUE;
        for(Instance ins:train){
            double x = measures[5].distance(wrongUn, ins);
            System.out.print(""+x);
            if(x<min1) {
                System.out.print(" NEW MIN ");
                min1 = x;
            }
            else if(x==min1)
                System.out.print("   TIE   ");
            else
                System.out.print("         ");

            double y = jayLCSS.distance(wrongUn, ins);
            System.out.print(","+y);
            if(y<min2) {
                System.out.print(" NEW MIN ");
                min2=y;
            }
            else if(y==min2)
                System.out.print("   TIE   ");
            else
                System.out.print("         ");
            if(x!=y)
                System.out.println(" MISMATCH");
            else
                System.out.println(" Class "+ins.classValue());
        }

        System.out.println(" new prediction = "+knn.classifyInstance(wrongUn)+"   k = "+        knn.getK());
        System.out.println(" new prediction = "+jayLCSS.classifyInstance(wrongUn));
        double[] dist1=knn.distributionForInstance(wrongUn);
        double[] dist2=jayLCSS.distributionForInstance(wrongUn);
        for(double d:dist1)
            System.out.print(d+",");
        System.out.println("");
        for(double d:dist2)
            System.out.print(d+",");

        int c1=countCorrect(knn,test);
        int c2=countCorrect(jayLCSS,test);
        System.out.println("\n George count = "+c1+" Jay count = "+c2);
        for(int i=0;i<test.numInstances();i++){
            double x= knn.classifyInstance(test.instance(i));
            System.out.print(i+","+x);
            double[] x2= knn.distributionForInstance(test.instance(i));
            for(double d: x2)
                System.out.print(","+d);

            System.out.print("\n");
        }

  */

    }

}
