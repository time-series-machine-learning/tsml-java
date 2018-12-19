package timeseriesweka.classifiers.ensembles.voting;

import java.util.Arrays;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Individuals vote based on their weight * (confidence^power). The power scales the 
 * relative differences between different confidences, effectively up-weighting those 
 * individuals that are more confident in their vote
 * 
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class MajorityVoteByPoweredConfidence extends ModuleVotingScheme {
    
    private double power = 2.0;
    
    public MajorityVoteByPoweredConfidence() {
        
    }
    
    public MajorityVoteByPoweredConfidence(double power) {
        this.power = power;
    }
    
    public MajorityVoteByPoweredConfidence(int numClasses) {
        this.numClasses = numClasses;
    }
    
    public MajorityVoteByPoweredConfidence(int numClasses, double power) {
        this.power = power;
        this.numClasses = numClasses;
    }

    public double getPower() {
        return power;
    }

    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        
        int pred;
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
            
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            Math.pow((modules[m].trainResults.getDistributionForInstance(trainInstanceIndex)[pred]), power);
        }
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
                
        int pred;
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].testResults.getPredClassValue(testInstanceIndex); 
            
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            Math.pow((modules[m].testResults.getDistributionForInstance(testInstanceIndex)[pred]), power);
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        int pred;
        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = modules[m].getClassifier().distributionForInstance(testInstance);
            storeModuleTestResult(modules[m], dist);
            
            pred = (int)indexOfMax(dist);
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            Math.pow(dist[pred], power);
        }
        
        return normalise(preds);
    }
    
}
