package ml6002b2022.week2_demo;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;

/**
 * Week 2 demo todo:
 * 1. Run through majority class classifier with iris and a discrete problem
 *      1.1 fit and predict on train
 *      1.2 Do first train/test split.
 *      1.3 Introduce Evaluation https://weka.sourceforge.io/doc.dev/index.html?weka/classifiers/Evaluation.html
 * 2. Run through histogram classifier
 * 3. Do an information gain example by hand
 * 4. Measure info gain in code (see lab sheet)
 * 5. Build a tree by hand
 * 6. Build an ID3 Tree in code
 */
public class Week2Examples {

    public static void simpleDemo(Classifier cls, Instances data) throws Exception {
        cls.buildClassifier(data);
        System.out.println(cls);
        int correct=0;
        for(Instance ins:data) {
            int pred = (int) cls.classifyInstance(ins);
            int actual = (int) ins.classValue();
            System.out.println(" Predicted  = "+pred);
            if (pred == actual)
                correct++;
//            double[] probs = mc.distributionForInstance(ins);
        }
        System.out.println(" Train set correct = "+correct+" Accuracy = "+(correct/(double)data.numInstances()));
        Instances[] split = InstanceTools.resampleInstances(data,0,0.5);
        Evaluation evaluation = new Evaluation(split[0]);
        evaluation.evaluateModel(cls, split[1]);
        System.out.println(" Evaluation  = "+evaluation.toSummaryString());
    }
    public static void discreteProblem() throws IOException {
        Instances train = DatasetLoading.loadData("src/main/java/ml6002b2022/week2_demo/Arsenal_TRAIN");
        Instances test = DatasetLoading.loadData("src/main/java/ml6002b2022/week2_demo/Arsenal_TEST");

    }

    public static void main(String[] args) throws Exception {
        Instances wdbc = DatasetLoading.loadData("src/main/java/ml6002b2022/week1_demo/wdbc");
        MajorityClassClassifier mc= new MajorityClassClassifier();
//        simpleDemo(mc,wdbc);
        HistogramClassifier hc = new HistogramClassifier();
        simpleDemo(hc, wdbc);
        discreteProblem();
    }



}
