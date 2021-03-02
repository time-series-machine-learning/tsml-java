package tests;

import core.contracts.Dataset;
import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.dictionary_based.TDE;
import tsml.classifiers.distance_based.proximity.ProximityForest;
import tsml.classifiers.interval_based.DrCIF;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Simple development class to road test the classifiers in HiveCote 2, and HC2 itself
 *
 */
public class ClassifierSanityChecks {

    static String[] classifiers={"DrCIF","TDE","PF","Arsenal","STC"};
    static EnhancedAbstractClassifier setClassifier(String str){
        //MNissing Arsenal and HC2

        EnhancedAbstractClassifier c = null;
        switch(str){
            case "DrCIF":
                c= new DrCIF();
                break;
            case "TDE":
                c= new TDE();
            break;
            case "PF":
                c=new ProximityForest();
                break;
            case "STC":
                c= new ShapeletTransformClassifier();
                break;

        }
        return c;
    }

    /**
     * To use before finalising tests
     */
    public static void hackFile() throws Exception {
        String path="src/main/java/experiments/data/tsc/";
        String problem="ArrowHead";

        Instances train= DatasetLoading.loadData(path+problem+"/"+problem+"_TRAIN.arff");
        Instances test= DatasetLoading.loadData(path+problem+"/"+problem+"_TEST.arff");

        for(String str:classifiers) {
            EnhancedAbstractClassifier c = setClassifier(str);
            if(c!=null) {
                c.buildClassifier(train);
                double acc = ClassifierTools.accuracy(test, c);
                System.out.println(str+" on "+problem+"  test acc = " + acc);
            }
        }



    }

    public static void main(String[] args) throws Exception {
        hackFile();

    }

}
