/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 *
 * TODO Currently should not be used, as cannot guarantee the range of the weights,
 * which should be 0 to 1. Was a random adaption from CEN just to try it out.
 * 
 * @author James
 */
public class ConfusionEntropyByClass extends ConfusionEntropy {
    
    public ConfusionEntropyByClass() {
        uniformWeighting = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
       double[] weights = new double[numClasses];
        for (int j = 0; j < numClasses; j++) 
            weights[j] = cen_j(module.trainResults.confusionMatrix, j, numClasses);

        return weights;
    }
    
}
