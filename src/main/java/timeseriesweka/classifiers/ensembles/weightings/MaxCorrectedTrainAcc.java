
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Sets each module's weighting to Math.max(0.01, module.trainResults.acc - maxClassWeighting),
 * where maxClassWeighting is the proportion of cases belonging to the most common class, 
 * i.e the accuracy expected from a completely biased classifier
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MaxCorrectedTrainAcc extends ModuleWeightingScheme {

    double maxClassWeighting;
    
    public MaxCorrectedTrainAcc() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        double[] classDist = new double[numClasses];
        for (int i = 0; i < modules[0].trainResults.getTrueClassVals().length; i++)
            classDist[(int)modules[0].trainResults.getTrueClassVals()[i]]++;
        
        maxClassWeighting = classDist[0];
        for (int i = 1; i < classDist.length; i++) 
            if (classDist[i] > maxClassWeighting)
                maxClassWeighting = classDist[i];
        
        for (EnsembleModule m : modules) //by default, sets weights independently for each module
            m.posteriorWeights = defineWeighting(m, numClasses);
        
        //some schemes may sets weights for each moduel relative to the rest, and 
        //so will need to override this method
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        //made non zero (effectively 1% accuracy) in weird case that all classifiers get less than expected acc
        return makeUniformWeighting(Math.max(0.01, module.trainResults.acc - maxClassWeighting), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
