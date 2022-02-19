package ml6002b2022.week3_demo;

import experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * Ensemble of C4.5 classifiers using simply majority vote
 *
 * 1. All with same train data
 * 2. Diversify on train data: sample instances or sample attributes
 */
public class TonyEnsemble extends AbstractClassifier {
    int numBaseClassifiers=100;
    ArrayList<Classifier> myEnsemble;



    @Override
    public Capabilities getCapabilities(){
        Capabilities result = new J48().getCapabilities();
        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        myEnsemble = new ArrayList<>(numBaseClassifiers);
        for(int i=0;i<numBaseClassifiers;i++){
            Classifier c = new J48();
            ((J48)c).setSeed(i);
            data.randomize(new Random());
            Instances temp = new Instances(data, 0,data.numInstances()/2);
            c.buildClassifier(temp);
            myEnsemble.add(c);
        }
    }

    public double classifyInstance(Instance ins) throws Exception {
        int numClassValues = ins.numClasses();
        int[] votes = new int[numClassValues];
        for(Classifier c:myEnsemble){
            int vote = (int) c.classifyInstance(ins);
            votes[vote]++;
        }
        //Get the ARG MAX
        int finalVote=0;
        for(int i=1;i<votes.length;i++){
            if(votes[i]>votes[finalVote])
                finalVote=i;
        }
        return finalVote;
    }

    public static void main(String[] args) throws Exception {
        Instances playGolf= DatasetLoading.loadData("src/main/java/ml6002b2022/week3_demo/playGolf");
        TonyEnsemble t= new TonyEnsemble();
        t.buildClassifier(playGolf);
        int pred = (int)t.classifyInstance(playGolf.instance(0));
        System.out.println(" Predict  ="+pred);

        Instances gunPointTrain= DatasetLoading.loadData("src/main/java/experiments/data/tsc/GunPoint/GunPoint_TRAIN");
        Instances gunPointTest= DatasetLoading.loadData("src/main/java/experiments/data/tsc/GunPoint/GunPoint_TEST");
        t.buildClassifier(gunPointTrain);
        int correct=0;
        for(Instance ins:gunPointTrain) {
            pred = (int) t.classifyInstance(gunPointTest.instance(0));
            if(pred == (int)ins.classValue())
                correct++;
        }
        System.out.println(" Number correct  =" + correct+" out of "+gunPointTrain.numInstances());
        correct=0;
        for(Instance ins:gunPointTest) {
            pred = (int) t.classifyInstance(gunPointTest.instance(0));
            if(pred == (int)ins.classValue())
                correct++;
        }
        System.out.println(" Number correct  =" + correct+" out of "+gunPointTest.numInstances());
    }


}
