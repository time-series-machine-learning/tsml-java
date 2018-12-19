package timeseriesweka.classifiers.ensembles.voting;

import java.util.Arrays;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Majority vote, however classifiers' vote is weighted by the confidence in their prediction,
 * i.e distForInst[pred]
 * 
 * @author James Large
 */
public class MajorityVoteByConfidence extends ModuleVotingScheme {
    
    public MajorityVoteByConfidence() {
    }
    
    public MajorityVoteByConfidence(int numClasses) {
        this.numClasses = numClasses;
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
                            modules[m].trainResults.getDistributionForInstance(trainInstanceIndex)[pred];
        }
        
        
//debug start        
        double[] unweightedPreds = new double[numClasses];
        
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
            unweightedPreds[pred]++;
        }
        
        for(int m = 0; m < modules.length; m++) {
            printlnDebug(modules[m].getModuleName() + " distForInst:  " + Arrays.toString(modules[m].trainResults.getDistributionForInstance(trainInstanceIndex)));
            printlnDebug(modules[m].getModuleName() + " priorweights: " + modules[m].priorWeight);
            printlnDebug(modules[m].getModuleName() + " postweights:  " + Arrays.toString(modules[m].posteriorWeights));
            printlnDebug(modules[m].getModuleName() + " voteweight:   " + (modules[m].priorWeight * 
                            modules[m].posteriorWeights[(int) modules[m].trainResults.getPredClassValue(trainInstanceIndex)] * 
                            modules[m].trainResults.getDistributionForInstance(trainInstanceIndex)[(int) modules[m].trainResults.getPredClassValue(trainInstanceIndex)]));
        }
        
        printlnDebug("Ensemble Votes: " + Arrays.toString(unweightedPreds));
        printlnDebug("Ensemble Dist:  " + Arrays.toString(preds));
        printlnDebug("Normed:         " + Arrays.toString(normalise(preds)));
        printlnDebug("");
//debug end
        
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
                            modules[m].testResults.getDistributionForInstance(testInstanceIndex)[pred];
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
                            dist[pred];
        }
        
        return normalise(preds);
    }
    
}
