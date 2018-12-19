package timeseriesweka.classifiers.ensembles.voting;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
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
        return modules[bestModule].trainResults.getDistributionForInstance(trainInstanceIndex);
    }

    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        return modules[bestModule].testResults.getDistributionForInstance(testInstanceIndex);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        return modules[bestModule].getClassifier().distributionForInstance(testInstance);
    }
    
}
