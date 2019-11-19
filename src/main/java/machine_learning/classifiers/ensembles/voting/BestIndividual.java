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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;
import weka.core.Instance;

/**
 *
 * @author James Large james.large@uea.ac.uk
 */
public abstract class BestIndividual extends ModuleVotingScheme {

    //list of all best modules this instance of 'BestIndividual' has 'been involved with'
    //purely for experimental/analysis purposes and lazyness
    //in a single hesca run, only one will be chosen (stored in the bestModule int),
    //but these will store the best potentially over mutliple folds on multiple datasets
    //just stored as simple list here, it's up to the experimental code to divide them up
    //by fold/dataset
    protected ArrayList<Integer> bestModulesInds = new ArrayList<>();
    protected ArrayList<String> bestModulesNames = new ArrayList<>();
    
    public int bestModule; //bestModule on this particular run of the ensemble
    
    public BestIndividual() {
    }
    
    public BestIndividual(int numClasses) {
        this.numClasses = numClasses;
    }

    public ArrayList<Integer> getBestModulesInds() {
        return bestModulesInds;
    }

    public ArrayList<String> getBestModulesNames() {
        return bestModulesNames;
    }
    
    /**
     * @return map<dset, bestclassifierperfold>
     * @throws Exception if (numBestIndsStored != dsets.length * folds) 
     */
    public Map<String, ArrayList<String>> splitBestIndividualString(String[] dsets, int numFolds) throws Exception {
        if (bestModulesNames.size() != dsets.length * numFolds)
            throw new Exception("not all folds present");
        
        Map<String, ArrayList<String>> res = new HashMap<>(dsets.length);
        
        int globalIndex = 0;
        for (int d = 0; d < dsets.length; d++) {
            ArrayList<String> fs = new ArrayList<>(numFolds);
            for (int f = 0; f < numFolds; f++)
                fs.add(bestModulesNames.get(globalIndex++));
           
            res.put(dsets[d], fs);
        }
        
        return res;
    }
    
    public Map<String, ArrayList<Integer>> splitBestIndividualIndex(String[] dsets, int numFolds) throws Exception {
        if (bestModulesNames.size() != dsets.length * numFolds)
            throw new Exception("not all folds present");
        
        Map<String, ArrayList<Integer>> res = new HashMap<>(dsets.length);
        
        int globalIndex = 0;
        for (int d = 0; d < dsets.length; d++) {
            ArrayList<Integer> fs = new ArrayList<>(numFolds);
            for (int f = 0; f < numFolds; f++)
                fs.add(bestModulesInds.get(globalIndex++));
           
            res.put(dsets[d], fs);
        }
        
        return res;
    }
    
    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        return modules[bestModule].trainResults.getProbabilityDistribution(trainInstanceIndex);
    }

    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        return modules[bestModule].testResults.getProbabilityDistribution(testInstanceIndex);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        return modules[bestModule].getClassifier().distributionForInstance(testInstance);
    }
    
}
