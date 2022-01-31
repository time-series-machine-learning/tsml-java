package ml6002b2022;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MyClassifier extends AbstractClassifier {
    double positiveMean;
    double negativeMean;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(data.numClasses()>2)
            throw new Exception("Can only handle two classes");
        positiveMean=negativeMean=0;
        int countP=0, countN=0;
        for(Instance ins:data){
            if(ins.classValue()==0) {
                negativeMean += ins.value(0);
                countN++;
            }
            else{
                positiveMean += ins.value(0);
                countP++;
            }
        }
        negativeMean/=countN;
        positiveMean/=countP;

    }

    @Override
    public double classifyInstance(Instance data){
        double x = data.value(0);
        double distToNeg = Math.abs(x-negativeMean);
        double distToPos = Math.abs(x-positiveMean);
        if(distToNeg<distToPos)
            return 0.0;
        return 1.0;
    }
    @Override
    public double[] distributionForInstance(Instance data) {
        double[] prob=new double[data.numClasses()];
        double x = data.value(0);
        double distToNeg = Math.abs(x-negativeMean);
        double distToPos = Math.abs(x-positiveMean);
        prob[0]=distToPos/(distToNeg+distToPos);
        prob[1]=distToNeg/(distToNeg+distToPos);
        return prob;
    }
}
