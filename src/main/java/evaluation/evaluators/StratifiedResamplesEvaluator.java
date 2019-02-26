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
package evaluation.evaluators;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class StratifiedResamplesEvaluator extends Evaluator {
    int numFolds;
    double propInstancesInTrain;

    private ClassifierResults[] resultsPerFold;
    
    public StratifiedResamplesEvaluator() {
        super(0,false,false);
        
        this.numFolds = 5;
        this.propInstancesInTrain = 0.5;
    }
    
    public StratifiedResamplesEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed,cloneData,setClassMissing);
        
        this.numFolds = 5;
        this.propInstancesInTrain = 0.5;
    }

    public double getPropInstancesInTrain() {
        return propInstancesInTrain;
    }

    public void setPropInstancesInTrain(double propInstancesInTrain) {
        this.propInstancesInTrain = propInstancesInTrain;
    }
    
    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }
    
    private ClassifierResults performLOOCVInstead(Classifier classifier, Instances dataset) throws Exception { 
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed,cloneData,setClassMissing);
        return cv.evaluate(classifier, dataset);
    }
    
    public ClassifierResults[] getResultsOfEachSample() {
        return resultsPerFold;
    }
    
    /**
     * This returns a single ClassifierResults object, however in common usage 
     * a ClassifierResults object would typically refer to only one of the resamples,
     * not all of them. The object returned by this method should effectively be treated
     * as a special case. The parameters line is not reliable, as optimal parameters 
     * are calculated 30 times, for example. 
     * 
     * What this will return is an object where the predictions are concatenated 
     * for each resample, i.e the predictions are ordered as all predictions for the 
     * test set of fold0, then the predictions for fold1, etc. 
     * 
     * Therefore, if you're evaluating multiple classifier on the same dataset using this 
     * evaluator, the predictions will all line up to each other (assuming they are seeded the
     * same to produce the same resamples). Each prediction will refer to the same test 
     * case being predicted after being trained on the same data. 
     * 
     * Additionally, concatenating the folds in this manner means that the stats 
     * reported calculated by ClassifierResults are automatically the stats averaged 
     * over the resamples
     * 
     * If you want to access the classifier resutls objects for each fold, these are 
     * also stored in this evaluator object, call getResultsOfEachSample()
     * 
     * @param classifier
     * @param dataset
     * @return
     * @throws Exception 
     */
    @Override
    public ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        
        //todo revisit, suppose numFolds = 30, propInTrain = 0.5, numInstances = 20, 20 choose 10 = 184756 >>>>> 30...
//        if (dataset.numInstances() <= numFolds) {
//            System.out.println("Warning, num resamples requested is greater than the number of instances, "
//                    + "performing a leave-one-out cross validation instead");
//            return performLOOCVInstead(classifier, dataset);
//        }
        
        if (cloneData)
            dataset = new Instances(dataset);

        resultsPerFold = new ClassifierResults[numFolds]; 
        ClassifierResults allFoldsResults = new ClassifierResults(dataset.numClasses());
        allFoldsResults.setTimeUnit(TimeUnit.NANOSECONDS);
        allFoldsResults.turnOffZeroTimingsErrors();
                
        for (int fold = 0; fold < numFolds; fold++) {
            Instances[] resampledData = InstanceTools.resampleInstances(dataset, seed, propInstancesInTrain);
            
            classifier.buildClassifier(resampledData[0]);
            resultsPerFold[fold] = new ClassifierResults(dataset.numClasses());
            resultsPerFold[fold].setTimeUnit(TimeUnit.NANOSECONDS);
            resultsPerFold[fold].turnOffZeroTimingsErrors();
            
            //todo, implement this loop via SingleTestSetEvluator            
            for (Instance testinst : resampledData[1]) {
                if (setClassMissing)
                    testinst.setClassMissing();
                
                long startTime = System.nanoTime();
                double[] dist = classifier.distributionForInstance(testinst);
                long predTime = System.nanoTime()- startTime;
                
                resultsPerFold[fold].addPrediction(testinst.classValue(), dist, indexOfMax(dist), predTime, "");
                allFoldsResults.addPrediction(testinst.classValue(), dist, indexOfMax(dist), predTime, "");
            }
            
            resultsPerFold[fold].turnOnZeroTimingsErrors();
            resultsPerFold[fold].findAllStatsOnce(); 
        }
   
        allFoldsResults.turnOnZeroTimingsErrors();
        allFoldsResults.findAllStatsOnce(); 
        return allFoldsResults;
    }
}

