package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Non uniform weighting scheme, calculates the recall for each class and uses those 
 * as the classifier's weighting
 * 
 * @author James Large
 */
public class RecallByClass extends ModuleWeightingScheme {

    public RecallByClass() {
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    @Override
    protected double[] defineWeighting(EnsembleModule module, int numClasses) {
        double[] weights = new double[numClasses];
        for (int c = 0; c < numClasses; c++) 
            weights[c] = computeRecall(module.trainResults.confusionMatrix, c);

        return weights;
    }
    
    protected double computeRecall(double[][] confMat, int c) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        if (tp == .0)
            return .0000001; 
        //some very small non-zero value, in the extreme case that no classifiers
        //in the entire ensemble classified cases of this class correctly
        //happened once on adiac (37 classes)
        
        double fn = 0.0;
        
        for (int i = 0; i < confMat.length; i++)
            if (i!=c)
                fn += confMat[c][i];
        
        double recall = tp / (tp+fn);
        return recall;
    }
    
}
