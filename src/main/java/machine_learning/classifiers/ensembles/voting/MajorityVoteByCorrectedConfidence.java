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

import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;
import static utilities.GenericTools.indexOfMax;
import weka.core.Instance;

/**
 * Individuals vote based on their weight * (confidence in prediction - 1/C), where
 * C is the number of classes. Subtracting 1/C effectively removes the aspect of random
 * guessing. In a 2 class case, if a classifier's distforinst is .9,.1, it is very confident
 * that the class is 0. However if its dist is .55,.45, it may as well just be guessing, however
 * the value .55 by itself does not reflect that, because the range of values is 0-1
 * 
 * The 'corrected' confidences would instead be in the range 0-0.5, and in the two cases above 
 * would then be .4 and .05. Thus this voting system disfavours more heavily those classifiers
 * that are unsure of their decision
 * 
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class MajorityVoteByCorrectedConfidence extends ModuleVotingScheme {
    
    public MajorityVoteByCorrectedConfidence() {
        
    }
    
    public MajorityVoteByCorrectedConfidence(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        
        double normValue = 1.0/numClasses; 
        
        int pred;
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
            
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            (modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)[pred] - normValue);
        }
        
        
//debug start        
//        double[] unweightedPreds = new double[numClasses];
//        
//        for(int m = 0; m < modules.length; m++){
//            pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
//            unweightedPreds[pred]++;
//        }
//        
//        for(int m = 0; m < modules.length; m++) {
//            printlnDebug(modules[m].getModuleName() + " distForInst:  " + Arrays.toString(modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)));
//            printlnDebug(modules[m].getModuleName() + " priorweights: " + modules[m].priorWeight);
//            printlnDebug(modules[m].getModuleName() + " postweights:  " + Arrays.toString(modules[m].posteriorWeights));
//            printlnDebug(modules[m].getModuleName() + " voteweight:   " + (modules[m].priorWeight * 
//                            modules[m].posteriorWeights[(int) modules[m].trainResults.getPredClassValue(trainInstanceIndex)] * 
//                            (modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)[(int) modules[m].trainResults.getPredClassValue(trainInstanceIndex)] - normValue)));
//        }
//        
//        printlnDebug("Ensemble Votes: " + Arrays.toString(unweightedPreds));
//        printlnDebug("Ensemble Dist:  " + Arrays.toString(preds));
//        printlnDebug("Normed:         " + Arrays.toString(normalise(preds)));
//        printlnDebug("");
//debug end
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        
        double normValue = 1.0/numClasses; 
        
        int pred;
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].testResults.getPredClassValue(testInstanceIndex); 
            
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            (modules[m].testResults.getProbabilityDistribution(testInstanceIndex)[pred] - normValue);
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        double normValue = 1.0/numClasses; 
        
        int pred;
        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = distributionForNewInstance(modules[m], testInstance);
            
            pred = (int)indexOfMax(dist);
            preds[pred] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[pred] * 
                            (dist[pred] - normValue);
        }
        
        return normalise(preds);
    }
    
}
