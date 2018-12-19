package papers;

import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;

public class Bagnall17Forests {
    
    public static void main(String[] args) throws Exception {
        buildRotFTimeEstimator();
    }
    
    public static void buildRotFTimeEstimator() throws Exception{
        LinearRegression linear=new LinearRegression();
        Classifier gp = new GaussianProcesses();
        Classifier smo =new SMOreg();
        Instances data=ClassifierTools.loadData("C:\\Research\\Papers\\Working Papers\\Forest Experiments\\RotFTimingExperiments\\RotFTimingReduced");
        Evaluation eval=new Evaluation(data);
        eval.crossValidateModel(linear, data,10,new Random());
        double m1=eval.errorRate();
        m1=Math.sqrt(m1);
        double m2=eval.meanAbsoluteError();
       linear.buildClassifier(data);
        System.out.println(" Full data model ="+linear);
        System.out.println("CV Linear regression MSE ="+m1+" MAE = "+m2);
        eval.crossValidateModel(gp, data,10,new Random());
        double m3=eval.errorRate();
        m3=Math.sqrt(m3);
        
        double m4=eval.meanAbsoluteError();
        System.out.println(" GP regression MSE ="+m3+" MAE = "+m4);
        eval.crossValidateModel(smo, data,10,new Random());
        double m5=eval.errorRate();
        m5=Math.sqrt(m5);
        double m6=eval.meanAbsoluteError();
        System.out.println(" SMO regression MSE ="+m5+" MAE = "+m6);
        
        
        
        
        
    }
    
    public static void buildRandRotF1SpeedUpEstimator(){
        
    }
    
}
