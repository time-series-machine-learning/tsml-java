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
 * Simple majority vote, gets the prediction of each module and adds it's weight
 * to that class' overall weight
 * 
 * @author James Large
 */
public class MajorityVote extends ModuleVotingScheme {

    public MajorityVote() {
    }
    
    public MajorityVote(int numClasses) {
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
        for(int c = 0; c < modules.length; c++){
            pred = (int) modules[c].trainResults.getPredClassValue(trainInstanceIndex); 
            
            preds[pred] += modules[c].priorWeight * 
                           modules[c].posteriorWeights[pred];
        }
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        
        int pred;
        for(int c = 0; c < modules.length; c++){
            pred = (int) modules[c].testResults.getPredClassValue(testInstanceIndex); 
            preds[pred] += modules[c].priorWeight * 
                           modules[c].posteriorWeights[pred];
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
                           modules[m].posteriorWeights[pred];
        }
        
        return normalise(preds);
    }
}
