/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package machine_learning.classifiers.ensembles.voting;

import java.util.Arrays;
import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;
import static utilities.GenericTools.indexOfMax;
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
                            modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)[pred];
        }
        
        
//debug start        
        double[] unweightedPreds = new double[numClasses];
        
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
            unweightedPreds[pred]++;
        }
        
        for(int m = 0; m < modules.length; m++) {
            printlnDebug(modules[m].getModuleName() + " distForInst:  " + Arrays.toString(modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)));
            printlnDebug(modules[m].getModuleName() + " priorweights: " + modules[m].priorWeight);
            printlnDebug(modules[m].getModuleName() + " postweights:  " + Arrays.toString(modules[m].posteriorWeights));
            printlnDebug(modules[m].getModuleName() + " voteweight:   " + (modules[m].priorWeight * 
                            modules[m].posteriorWeights[(int) modules[m].trainResults.getPredClassValue(trainInstanceIndex)] * 
                            modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)[(int) modules[m].trainResults.getPredClassValue(trainInstanceIndex)]));
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
                            modules[m].testResults.getProbabilityDistribution(testInstanceIndex)[pred];
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        int pred;
        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = distributionForNewInstance(modules[m], testInstance);
            
            pred = (int)indexOfMax(dist);
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            dist[pred];
        }
        
        return normalise(preds);
    }
    
}
