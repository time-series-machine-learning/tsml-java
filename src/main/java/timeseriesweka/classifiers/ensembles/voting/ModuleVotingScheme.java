
package timeseriesweka.classifiers.ensembles.voting;

import java.util.ArrayList;
import java.util.Random;
import utilities.DebugPrinting;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import utilities.ClassifierResults;
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
    
    public double indexOfMax(double[] dist) {
        double max = dist[0];
        double maxInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        return maxInd;
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
    
    public void storeModuleTestResult(EnsembleModule module, double[] dist) {
        if (module.testResults == null)
            module.testResults = new ClassifierResults();
        
        module.testResults.storeSingleResult(dist);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
