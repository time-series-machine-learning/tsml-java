package timeseriesweka.classifiers.ensembles.voting;

import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Uses the weighted confidences of each module that the instance is in EACH class (not just the most likely)
 * 
 * i.e in a 2-class problem, a module's distforinst maybe be .6 / .4, 
 *      .6 * weight_c1 will be added to class 1
 *      .4 * weight_c2 will be added to class 2 as well
 * 
 * @author James Large
 */
public class MajorityConfidence extends ModuleVotingScheme {
    
    public MajorityConfidence() {
    }
    
    public MajorityConfidence(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        
        for(int m = 0; m < modules.length; m++){
            double[] p=modules[m].trainResults.getDistributionForInstance(trainInstanceIndex);
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[c] * p[c];
            }
        }
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        
        for(int m = 0; m < modules.length; m++){
            double[] p=modules[m].testResults.getDistributionForInstance(testInstanceIndex);
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[c] * p[c];
            }
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = modules[m].getClassifier().distributionForInstance(testInstance);
            storeModuleTestResult(modules[m], dist);
            
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[c] * 
                            dist[c];
            }
        }
        
        return normalise(preds);
    }
    
}
