package utilities;

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
}
