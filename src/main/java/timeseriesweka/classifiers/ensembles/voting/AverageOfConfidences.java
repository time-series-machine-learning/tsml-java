package timeseriesweka.classifiers.ensembles.voting;

import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Each class' probability is defined as the average of each classifier's weighted 
 * confidence that the instance is of this class
 *
 * Afterthought: should actually give identical results to MajorityConfidence, it's
 * just the sum (as in majority) divided by some constant
 * 
 * @author James Large
 */
public class AverageOfConfidences extends ModuleVotingScheme {
    public AverageOfConfidences() {
        
    }
    
    public AverageOfConfidences(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        
        for (int c = 0; c < numClasses; c++) {
            double sum = .0;
            for(int m = 0; m < modules.length; m++){
                double[] p=modules[m].trainResults.getDistributionForInstance(trainInstanceIndex);
                sum += modules[m].priorWeight * 
                        modules[m].posteriorWeights[c]*p[c];
            }
            preds[c] = sum/modules.length;
        }
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            double sum = .0;
            for(int m = 0; m < modules.length; m++){
                double[] p=modules[m].testResults.getDistributionForInstance(testInstanceIndex);
                sum += modules[m].priorWeight * 
                        modules[m].posteriorWeights[c]*p[c];
            }
            preds[c] = sum/modules.length;
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        double[][] dists = new double[modules.length][];
        for(int m = 0; m < modules.length; m++){
            dists[m] = modules[m].getClassifier().distributionForInstance(testInstance);
            storeModuleTestResult(modules[m], dists[m]);
        }
         
        for (int c = 0; c < numClasses; c++) {
            double sum = .0;
            for(int m = 0; m < modules.length; m++){
                sum += modules[m].priorWeight * 
                        modules[m].posteriorWeights[c] * 
                        dists[m][c];
            }
            preds[c] = sum/modules.length;
        }
        
        return normalise(preds);
    }
}
