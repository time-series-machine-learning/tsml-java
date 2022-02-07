package ml6002b2022.week2_demo;

import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MajorityClassClassifier extends AbstractClassifier {
    int[] count;
    double[] classDistribution;
    @Override
    public void buildClassifier(Instances data) throws Exception {
        count = new int[data.numClasses()];
        for(Instance ins:data){
            int c=(int)ins.classValue();
            count[c]++;
        }
        classDistribution= new double[data.numClasses()];
        for(int i=0;i<data.numClasses();i++)
            classDistribution[i]=count[i]/(double)data.numInstances();
    }
    @Override
    public double[] distributionForInstance(Instance ins){
        return classDistribution;
    }
    public String toString(){
        String str= "Class Distribution  = ";
        for(double d:classDistribution)
            str+=d+",";
        return str;
    }






}