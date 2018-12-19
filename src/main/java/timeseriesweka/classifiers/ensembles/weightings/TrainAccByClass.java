/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Simply calculates this classifier's accuracy on each class
 * 
 * @author James Large
 */
public class TrainAccByClass extends ModuleWeightingScheme {
    
    public TrainAccByClass() {
        uniformWeighting = false;
    }

    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        double[] weights = new double[numClasses];
        for (int c = 0; c < numClasses; c++) 
            weights[c] = calcClassAcc(module.trainResults.confusionMatrix, c);

        return weights;
    }
    
    protected double calcClassAcc(double [][] confMat, int c) {
        double correct = confMat[c][c];
        double wrong = .0;
        for (int i = 0; i < confMat.length; i++)
            if (i!=c)
                wrong += confMat[c][i];
        
        return correct / (wrong+correct);
    }
    
}
