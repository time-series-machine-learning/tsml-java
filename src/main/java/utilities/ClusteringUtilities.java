package utilities;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

public class ClusteringUtilities {
    public static double randIndex(int[] predicted, int[] actual){
        double A = 0, B = 0, C = 0, D = 0;

        for (int i = 0; i < predicted.length; i++){
            for (int n = 0; n < actual.length; n++){
                if ((predicted[i] == predicted[n]) && (actual[i] == actual[n])){
                    A++;
                }
                else if ((predicted[i] != predicted[n]) && (actual[i] != actual[n])){
                    B++;
                }
                else if ((predicted[i] == predicted[n]) && (actual[i] != actual[n])){
                    C++;
                }
                else{
                    D++;
                }
            }
        }

        return (A + B)/(A + B + C + D);
    }

    public static double randIndex(int[] predicted, Instances inst){
        double[] actual = inst.attributeToDoubleArray(inst.classIndex());

        double A = 0, B = 0, C = 0, D = 0;

        for (int i = 0; i < predicted.length; i++){
            for (int n = 0; n < actual.length; n++){
                if ((predicted[i] == predicted[n]) && (actual[i] == actual[n])){
                    A++;
                }
                else if ((predicted[i] != predicted[n]) && (actual[i] != actual[n])){
                    B++;
                }
                else if ((predicted[i] == predicted[n]) && (actual[i] != actual[n])){
                    C++;
                }
                else{
                    D++;
                }
            }
        }

        return (A + B)/(A + B + C + D);
    }

    public static void zNormalise(Instances data) {
        for (Instance inst: data){
            zNormalise(inst);
        }
    }

    public static void zNormalise(Instance inst){
        double meanSum = 0;
        int length = inst.numAttributes();

        if (length < 2) return;

        for (int i = 0; i < length; i++){
            meanSum += inst.value(i);
        }

        double mean = meanSum / length;

        double squareSum = 0;

        for (int i = 0; i < length; i++){
            double temp = inst.value(i) - mean;
            squareSum += temp * temp;
        }

        double stdev = Math.sqrt(squareSum/(length-1));

        if (stdev == 0){
            stdev = 1;
        }

        for (int i = 0; i < length; i++){
            inst.setValue(i, (inst.value(i) - mean) / stdev);
        }
    }

    public static void zNormalise(double[] inst){
        double meanSum = 0;

        for (int i = 0; i < inst.length; i++){
            meanSum += inst[i];
        }

        double mean = meanSum / inst.length;

        double squareSum = 0;

        for (int i = 0; i < inst.length; i++){
            double temp = inst[i] - mean;
            squareSum += temp * temp;
        }

        double stdev = Math.sqrt(squareSum/(inst.length-1));

        if (stdev == 0){
            stdev = 1;
        }

        for (int i = 0; i < inst.length; i++){
            inst[i] = (inst[i] - mean) / stdev;
        }
    }

    public static void zNormaliseWithClass(Instances data) {
        for (Instance inst: data){
            zNormaliseWithClass(inst);
        }
    }

    public static void zNormaliseWithClass(Instance inst){
        double meanSum = 0;
        int length = inst.numAttributes()-1;

        for (int i = 0; i < inst.numAttributes(); i++){
            if (inst.classIndex() != i) {
                meanSum += inst.value(i);
            }
        }

        double mean = meanSum / length;

        double squareSum = 0;

        for (int i = 0; i < inst.numAttributes(); i++){
            if (inst.classIndex() != i) {
                double temp = inst.value(i) - mean;
                squareSum += temp * temp;
            }
        }

        double stdev = Math.sqrt(squareSum/(length-1));

        if (stdev == 0){
            stdev = 1;
        }

        for (int i = 0; i < inst.numAttributes(); i++){
            if (inst.classIndex() != i) {
                inst.setValue(i, (inst.value(i) - mean) / stdev);
            }
        }
    }

    //Create lower half distance matrix.
    public static double[][] createDistanceMatrix(Instances data, DistanceFunction distFunc){
        double[][] distMatrix = new double[data.numInstances()][];
        distFunc.setInstances(data);

        for (int i = 0; i < data.numInstances(); i++){
            distMatrix[i] = new double[i+1];
            Instance first = data.get(i);

            for (int n = 0; n < i; n++){
                distMatrix[i][n] = distFunc.distance(first, data.get(n));
            }
        }

        return distMatrix;
    }
}
