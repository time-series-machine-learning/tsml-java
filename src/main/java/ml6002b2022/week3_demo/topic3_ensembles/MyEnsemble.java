package ml6002b2022.week3_demo.topic3_ensembles;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class MyEnsemble extends AbstractClassifier {
    ArrayList<Classifier> ensemble;
    int numClassifiers= 100;


    public MyEnsemble(){

    }
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        ensemble = new ArrayList<>();
        for(int i=0;i<numClassifiers;i++){
            Classifier c = new J48();
            instances.randomize(new Random());
            Instances tr=new Instances(instances,0,instances.numInstances()/2);
            c.buildClassifier(tr);
            ensemble.add(c);
        }
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception {
//Majority vote
        int[] counts= new int[ins.numClasses()];
        for(Classifier c:ensemble){
            int vote = (int)c.classifyInstance(ins);
            counts[vote]++;
        }
        int argMax=0;
        for(int i=1;i<counts.length;i++)
            if(counts[i]>counts[argMax])
                argMax=i;
        return argMax;
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        double[] probs = new double[inst.numClasses()];
        for(Classifier c:ensemble){
            double[] d = c.distributionForInstance(inst);
            for(int i=0;i<d.length;i++)
                probs[i]+=d[i];
        }
        double sum=0;
        for(int i=0;i<probs.length;i++)
            sum+=probs[i];
        for(int i=0;i<probs.length;i++)
            probs[i]/=sum;
        return probs;

    }
}
