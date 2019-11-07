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
                            Math.pow((modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex)[pred]), power);
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
                            Math.pow((modules[m].testResults.getProbabilityDistribution(testInstanceIndex)[pred]), power);
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
                            Math.pow(dist[pred], power);
        }
        
        return normalise(preds);
    }
    
}
