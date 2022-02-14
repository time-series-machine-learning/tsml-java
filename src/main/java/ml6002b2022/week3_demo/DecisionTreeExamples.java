package ml6002b2022.week3_demo;

import experiments.data.DatasetLoading;
import org.graalvm.compiler.core.common.type.ArithmeticOpTable;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;

/**
 * Demo plan
 *
 * Decision Trees:
 * Load data sets with all discrete (lab sheet example) and all continuous (Iris) attributes
 * Show capabilities of J48 and ID3
 * Built classifiers and print output
 * IG example example, show calculations in code, use on hand example
 * Switch to scikit learn and demo DecisionTreeClassifier
 * Cover some examples of parameters for J48
 *
 * Ensembles:
 * Build an ensemble from scratch
 * Demonstrate train/test splits
 */


public class DecisionTreeExamples {

    public static void capabilitiesExample() throws Exception {
        J48 c45 = new J48();
        Id3 id3 = new Id3();
        System.out.println(" Base class capabilities = "+ new DummyClassifier().getCapabilities());
        System.out.println(" C45 capabilities = "+c45.getCapabilities());
        System.out.println(" ID3 capabilities = "+id3.getCapabilities());
        Instances iris= DatasetLoading.loadData("src/main/java/experiments/data/uci/iris/iris");
        Instances playGolf= DatasetLoading.loadData("src/main/java/ml6002b2022/week3_demo/playGolf");

    }
    public static void  IGExample() throws IOException {
// Form a count matrix for temperature and outlook
        Instances playGolf= DatasetLoading.loadData("src/main/java/ml6002b2022/week3_demo/playGolf");
        int[][] outlook = new int[playGolf.attribute("Outlook").numValues()][playGolf.numClasses()];
        for(Instance ins:playGolf){
            outlook[(int)ins.value(0)][(int)ins.classValue()]++;
        }
        for(int[] x:outlook) {
            for (int y : x)
                System.out.print(y + ",");
            System.out.print("\n");
        }
        //Play golf counts: Each row is an attribute value, each column a class value
        double[][] t= {{2,2},{7,3}};
        double[][] o= {{2,3},{0,4},{3,2}};
        double[][] h=  {{5,4},{4,1}};
        double[][] w=  {{6,2},{3,3}};
        InfoGainSplitCrit infoGain = new InfoGainSplitCrit();
        Distribution dist= new Distribution(o);
        System.out.println(" Dist for outlook= "+dist.dumpDistribution());
        double outlookIG = infoGain.splitCritValue(dist);
        System.out.println(" Outlook IG = "+outlookIG);


    }

    public static void playGolfExample(){

    }
    public static void main(String[] args) throws Exception {
        capabilitiesExample();
        IGExample();

    }

}
