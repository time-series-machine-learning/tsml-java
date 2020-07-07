package tsml.contrib;

import experiments.data.DatasetLoading;
import statistics.simulators.DictionaryModel;
import weka.core.Instance;
import weka.core.Instances;

public class main {
    public static void main(String [] args) throws Exception {

        Instances[] data = DatasetLoading.sampleItalyPowerDemand(0);

        ShapeDTW1NN sdtw = new ShapeDTW1NN(30,null);
        sdtw.buildClassifier(data[0]);
        System.out.println(calculateAccuracy(data[1],sdtw));
    }

    public static double calculateAccuracy(Instances test, ShapeDTW1NN sdtw) throws Exception{
        double [] classValues = new double [test.numInstances()];
        for(int i=0;i<test.numInstances();i++) {
            classValues[i] = test.get(i).classValue();
        }
        double [] predicted = new double [test.numInstances()];
        for(int i=0;i<test.numInstances();i++) {
            predicted[i] = sdtw.classifyInstance(test.get(i));
        }
        int count = 0;
        for(int i=0;i<predicted.length;i++) {
            if(predicted[i] == classValues[i]) {
                count = count + 1;
            }
        }
        return (double) count/ (double) predicted.length;
    }
}
