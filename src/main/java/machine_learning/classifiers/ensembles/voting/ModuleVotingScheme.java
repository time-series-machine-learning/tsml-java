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

import utilities.DebugPrinting;
import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;
import weka.core.Instance;

/**
 * Base class for methods on combining ensemble members' ouputs into a single classification/distribution
 * 
 * @author James Large
 */
public abstract class ModuleVotingScheme implements DebugPrinting {
    
    protected int numClasses;  
    public boolean needTrainPreds = false;
    
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        this.numClasses = numClasses;
    }
    
    public abstract double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex)  throws Exception;
    
    public double classifyTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) throws Exception {
        double[] dist = distributionForTrainInstance(modules, trainInstanceIndex);
        return indexOfMax(dist);
    }
    
    public abstract double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex)  throws Exception;
    
    public double classifyTestInstance(EnsembleModule[] modules, int testInstanceIndex) throws Exception {
        double[] dist = distributionForTestInstance(modules, testInstanceIndex);
        return indexOfMax(dist);
    }
    
    public abstract double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception;
    
    public double classifyInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] dist = distributionForInstance(modules, testInstance);
        return indexOfMax(dist);
    }
    
    
//    protected static double indexOfMax(double[] dist) throws Exception {  
//        double  bsfWeight = -(Double.MAX_VALUE);
//        ArrayList<Integer>  bsfClassVals = null;
//        
//        for (int c = 0; c < dist.length; c++) {
//            if(dist[c] > bsfWeight){
//                bsfWeight = dist[c];
//                bsfClassVals = new ArrayList<>();
//                bsfClassVals.add(c);
//            }else if(dist[c] == bsfWeight){
//                bsfClassVals.add(c);
//            }
//        }
//
//        if(bsfClassVals == null)
//            throw new Exception("bsfClassVals == null, NaN problem");
//
//        double pred; 
//        //if there's a tie for highest voted class after all module have voted, settle randomly
//        if(bsfClassVals.size()>1)
//            pred = bsfClassVals.get(new Random(0).nextInt(bsfClassVals.size()));
//        else
//            pred = bsfClassVals.get(0);
//        
//        return pred;
//    }
    
    /**
     * makes array sum to 1
     */
    public double[] normalise(double[] dist) {
        //normalise so all sum to one 
        double sum=dist[0];
        for(int i = 1; i < dist.length; i++)
            sum += dist[i];
        
        if (sum == 0.0)
            for(int i = 0; i < dist.length; i++)
                dist[i] = 1.0/dist.length;
        else
            for(int i = 0; i < dist.length; i++)
                dist[i] /= sum;
        
        return dist;
    }
    
    protected double[] distributionForNewInstance(EnsembleModule module, Instance inst) throws Exception {
        long startTime = System.nanoTime();
        double[] dist = module.getClassifier().distributionForInstance(inst);
        long predTime = System.nanoTime() - startTime;

        storeModuleTestResult(module, dist, predTime);
        
        return dist;
    }
    
    public void storeModuleTestResult(EnsembleModule module, double[] dist, long predTime) throws Exception {
        if (module.testResults == null) {
            module.testResults = new ClassifierResults();   
            module.testResults.setTimeUnit(TimeUnit.NANOSECONDS);
            module.testResults.setBuildTime(module.trainResults.getBuildTime());
        }
        
        module.testResults.addPrediction(dist, indexOfMax(dist), predTime, "");
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
