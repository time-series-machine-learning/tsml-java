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
 * TODO what if there's tie for best? UNTESTED
 * 
 * The ensemble's distribution for an instance is equal to the single 'best' individual,
 * as defined by THEIR TEST ACCURACY. Results must have been read from file (i.e test preds
 * already exist at train time) Weighting scheme is irrelevant, only considers accuracy.
 * 
 * Mostly just written so that I can do the best individual within the existing framework for 
 * later testing
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class BestIndividualOracle extends BestIndividual {

    
    public BestIndividualOracle() {
        super();
    }
    
    public BestIndividualOracle(int numClasses) {
        super(numClasses);
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        super.trainVotingScheme(modules, numClasses);
        
        double bestAcc = -1;
        for (int m = 0; m < modules.length; ++m) {         
            if (modules[m].testResults.getAcc() > bestAcc) {
                bestAcc = modules[m].testResults.getAcc();
                bestModule = m;
            }
        }
        
        bestModulesInds.add(bestModule);
        bestModulesNames.add(modules[bestModule].getModuleName());
        
        printlnDebug(modules[bestModule].getModuleName());
    }
}
