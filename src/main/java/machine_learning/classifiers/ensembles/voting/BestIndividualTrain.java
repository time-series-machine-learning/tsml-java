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

/**
 *
 * TODO what if there's tie for best?  
 * 
 * The ensemble's distribution for an instance is equal to the single 'best' individual,
 * as defined by whatever (uniform) weighting scheme is being used. 
 * 
 * Mostly just written so that I can do the best individual within the existing framework for 
 * later testing
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class BestIndividualTrain extends BestIndividual {
    
    public BestIndividualTrain() {
        super();
    }
    
    public BestIndividualTrain(int numClasses) {
        super(numClasses);
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        super.trainVotingScheme(modules, numClasses);
        
        double bestWeight = -1;
        for (int m = 0; m < modules.length; ++m) {
            
            //checking that the weights are uniform
            double prevWeight = modules[m].posteriorWeights[0];
            for (int c = 1; c < numClasses; ++c)  {
                if (prevWeight == modules[m].posteriorWeights[c])
                    prevWeight = modules[m].posteriorWeights[c];
                else 
                    throw new Exception("BestIndividualTrain cannot be used with non-uniform weighting schemes");
            }
            
            if (modules[m].posteriorWeights[0] > bestWeight) {
                bestWeight = modules[m].posteriorWeights[0];
                bestModule = m;
            }
        }
        
        bestModulesInds.add(bestModule);
        bestModulesNames.add(modules[bestModule].getModuleName());
        
        printlnDebug(modules[bestModule].getModuleName());
    }
    
}
