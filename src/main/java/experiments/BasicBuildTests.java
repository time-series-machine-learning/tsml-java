/*
Class to do basic build tests for all classifiers
 */
package experiments;

import experiments.data.DatasetLoading;
import tsml.transformers.Transformer;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Does basic sanity check builds for all listed classifiers and transformers. Does not guarantee correctness,
 * just checks they all build and produce output
 *
 * @author ajb
 */
public class BasicBuildTests {


    public static void buildAllClassifiers(String[] problems, String[] classifiers, String path) {
        for(String str:problems){
            System.out.println("Building all for problem "+str);
            Instances train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
            Instances test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
            for(String cls:classifiers){
                System.out.print("\t Building "+cls+" .... ");
                Classifier c= ClassifierLists.setClassifierClassic(cls,0);
                try{
                    c.buildClassifier(train);
                    System.out.print("Built successfully. Accuracy = ");
                    double a=ClassifierTools.accuracy(test, c);
                    System.out.println(a);
                }catch(Exception e){
                    System.out.println("Classifier failed to build with exception "+e);
//                    e.printStackTrace();
                }
            }
        }
    }
    public static void buildAllTransforms(String[] problems, String[] transforms, String path) {
        for(String str:problems){
            System.out.println("Transforming all all for problem "+str);
            Instances train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
            Instances test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
            for(String trans:transforms){
                System.out.print("\t Building "+trans+" .... ");
                Transformer f = TransformLists.setClassicTransform(trans,0);
                try{
                    Instances trainTrans=f.transform(train);
                    System.out.print("\tTrain transformed successfully. Prior to Trans length = "+(train.numAttributes()-1));
                    Instances testTrans=f.transform(test);
                    System.out.println("\t\t   Test transformed successfully. Length = "+(testTrans.numAttributes()-1));
                }catch(Exception e){
                    System.out.println("Transform failed to build with exception "+e);
                    e.printStackTrace();
                    System.exit(0);
                }
            }
        }
    }



    public static void main(String[] args)  {
        System.out.println("Testing all SimpleBatch filters do not crash");
        String dataPath="src\\main\\java\\experiments\\data\\tsc\\";
        String[] problems={"ItalyPowerDemand","Chinatown","Beef"};
        String[] transforms=TransformLists.allFilters;
        buildAllTransforms(problems,transforms,dataPath);


        System.out.println("Testing core functionality of all TSC classifiers");
        String[] classifiers=ClassifierLists.allUnivariate;
        buildAllClassifiers(problems,classifiers,dataPath);
    }
}
