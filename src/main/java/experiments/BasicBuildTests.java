/*
Class to do basic build tests for all classifiers
 */
package experiments;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
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
            System.out.println("Building all for problem "+str);
            Instances train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
            Instances test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
            for(String trans:transforms){
                System.out.print("\t Building "+trans+" .... ");
                SimpleBatchFilter f = TransformLists.setClassicTransform(trans,0);
                try{
                    Instances trainTrans=f.process(train);
                    System.out.print("Train transformed successfully. Length = "+(trainTrans.numAttributes()-1));
                    Instances testTrans=f.process(test);
                    System.out.print("Test transformed successfully. Length = "+(testTrans.numAttributes()-1));
                }catch(Exception e){
                    System.out.println("Transform failed to build with exception "+e);
//                    e.printStackTrace();
                }
            }
        }
    }


    public static void main(String[] args)  {
        System.out.println("Testing core functionality of all TSC classifiers");
        String dataPath="src\\main\\java\\experiments\\data\\tsc\\";
        String[] problems={"Chinatown","ItalyPowerDemand","Beef"};
        String[] classifiers=ClassifierLists.allClassifiers;
        buildAllClassifiers(problems,classifiers,dataPath);
    }
}
