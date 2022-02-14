package ml6002b2022.week3_demo.topic3_ensembles;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class TonyEnsemble extends AbstractClassifier {
    ArrayList<Classifier> ensemble;
    int numClassifiers=100;

    /**
     * build an ensemble from scratch, without using any built in
     * tools. Implement an ensemble classifier that contains an array of J48 base
     * classifiers. Diversify your ensemble by sampling 50% of the train data for each
     * classifier (without replacement). Classify new instances with a simple majority
     *
     * @param data set of instances serving as training data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        ensemble = new ArrayList<>();
        for(int i=0;i<numClassifiers;i++){
            Classifier c=new J48();
            ensemble.add(c);
        }
        for(Classifier c:ensemble){
//Split the data: use tools or do it manually.
            data.randomize(new Random());
            Instances train=new Instances(data,0,data.numInstances()/2);
            c.buildClassifier(train);
        }
    }
    public double classifyInstance(Instance inst) throws Exception{
        int[] counts=new int[inst.numClasses()];
        for(Classifier c:ensemble){
            counts[(int)c.classifyInstance(inst)]++;
        }
        int argMax=0;
        for(int i=1;i<counts.length;i++)
            if(counts[i]>counts[argMax])
                argMax=i;
        return argMax;
    }

}
