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

import evaluation.storage.ClassifierResults;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An evaluator that performs k stratified random resamples (default k=30) of the given 
 * data and evaluates the given classifier(s) on each resample. 
 * 
 * Concatenated predictions across all resamples are returned from the main 
 * evaluate method, however predictions split across each resample can also be retrieved
 * afterwards
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class StratifiedResamplesEvaluator extends MultiSamplingEvaluator {
    double propInstancesInTrain;
    
    /**
     * If true, the seeds used to generate each resample shall simply be id 
     * of the resample in the loop, i.e. the values 0 to numFolds-1
     * 
     * This would mirror the generation of arff folds in Experiments, for example. 
     * This also means that the seed of this StratifiedResamplesEvaluator object
     * has no real use, aside from it would be stored as the fold id in the meta data
     * of the concatenated results object. 
     * 
     * Otherwise if false, the data resample seeds shall be randomly generated 
     * via the seed of this object. So still reproducable, but likely not aligned with  
     * resamples produced semi manually by just looping over numFolds and using i 
     * as the seed
     */
    boolean useEachResampleIdAsSeed;
    
    public StratifiedResamplesEvaluator() {
        super(0,false,false,false,false);
        
        this.numFolds = 30;
        this.propInstancesInTrain = 0.5;
    }
    
    public StratifiedResamplesEvaluator(int seed, boolean cloneData, boolean setClassMissing, boolean cloneClassifiers, boolean maintainClassifiers) {
        super(seed,cloneData,setClassMissing, cloneClassifiers, maintainClassifiers);
        
        this.numFolds = 30;
        this.propInstancesInTrain = 0.5;
    }
    
    /**
     * If true, the seeds used to generate each resample shall simply be id 
     * of the resample in the loop, i.e. the values 0 to numFolds-1
     * 
     * This would mirror the generation of arff folds in Experiments, for example. 
     * This also means that the seed of this StratifiedResamplesEvaluator object
     * has no real use, aside from it would be stored as the fold id in the meta data
     * of the concatenated results object. 
     * 
     * Otherwise if false, the data resample seeds shall be randomly generated 
     * via the seed of this object. So still reproducable, but likely not aligned with  
     * resamples produced semi manually by just looping over numFolds and using i 
     * as the seed
     */
    public boolean getUseEachResampleIdAsSeed() { 
        return useEachResampleIdAsSeed;
    }
        
    /**
     * If true, the seeds used to generate each resample shall simply be id 
     * of the resample in the loop, i.e. the values 0 to numFolds-1
     * 
     * This would mirror the generation of arff folds in Experiments, for example. 
     * This also means that the seed of this StratifiedResamplesEvaluator object
     * has no real use, aside from it would be stored as the fold id in the meta data
     * of the concatenated results object. 
     * 
     * Otherwise if false, the data resample seeds shall be randomly generated 
     * via the seed of this object. So still reproducable, but likely not aligned with  
     * resamples produced semi manually by just looping over numFolds and using i 
     * as the seed
     */
    public void setUseEachResampleIdAsSeed(boolean useEachResampleIdAsSeed) { 
        this.useEachResampleIdAsSeed = useEachResampleIdAsSeed;
    }
    
    public double getPropInstancesInTrain() {
        return propInstancesInTrain;
    }

    public void setPropInstancesInTrain(double propInstancesInTrain) {
        this.propInstancesInTrain = propInstancesInTrain;
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
     * Therefore, if you're evaluating multiple classifiers on the same dataset using this 
     * evaluator, the predictions will all line up to each other (assuming they are seeded the
     * same to produce the same resamples). Each prediction will refer to the same test 
     * case being predicted after being trained on the same data. 
     * 
     * Additionally, concatenating the folds in this manner means that the stats 
     * reported calculated by ClassifierResults are automatically the stats averaged 
     * over the resamples
     * 
     * If you want to access the classifier results objects for each fold, these are 
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
        
        ClassifierResults res = stratifiedResampleWithStats(classifier, dataset);
        res.findAllStatsOnce();
        return res;
    }
    
    public ClassifierResults stratifiedResampleWithStats(Classifier classifier, Instances dataset) throws Exception {
        return stratifiedResampleWithStats(new Classifier[] { classifier }, dataset)[0];
    }
    
    public ClassifierResults[] stratifiedResampleWithStats(Classifier[] classifiers, Instances dataset) throws Exception {
        if (cloneData)
            dataset = new Instances(dataset);
        if (cloneClassifiers)
            cloneClassifiers(classifiers);
        
        resultsPerFold = new ClassifierResults[classifiers.length][numFolds];
        
        ClassifierResults[] allConcatenatedClassifierRes = new ClassifierResults[classifiers.length] ;
        
        for (int classifierIndex = 0; classifierIndex < classifiers.length; ++classifierIndex) {
            
            //rebuild for each classifier so resamples are aligned
            //ignored if useEachResampleIdAsSeed == true
            Random classifierRng = new Random(seed); 
            
            long estimateTimeStart = System.nanoTime();
            
            for (int fold = 0; fold < numFolds; fold++) {
                Classifier foldClassifier = classifiers[classifierIndex];
                if (cloneClassifiers)
                    //use the clone instead
                    foldClassifier = foldClassifiers[classifierIndex][fold];
                
                int resampleSeed = useEachResampleIdAsSeed ? fold : classifierRng.nextInt();
                    
                SingleSampleEvaluator eval = new SingleSampleEvaluator(resampleSeed, this.cloneData, this.setClassMissing);
                resultsPerFold[classifierIndex][fold] = eval.evaluate(foldClassifier, dataset);
            }
            
            long estimateTime = System.nanoTime() - estimateTimeStart;
            
            ClassifierResults concatenatedClassifierRes = ClassifierResults.concatenateClassifierResults(resultsPerFold[classifierIndex]);
            concatenatedClassifierRes.setTimeUnit(TimeUnit.NANOSECONDS);
            concatenatedClassifierRes.setClassifierName(classifiers[classifierIndex].getClass().getSimpleName());
            concatenatedClassifierRes.setDatasetName(dataset.relationName());
            concatenatedClassifierRes.setFoldID(seed);
            concatenatedClassifierRes.setSplit("train"); //todo revisit, or leave with the assumption that calling method will set this to test when needed
            
            concatenatedClassifierRes.setErrorEstimateTime(estimateTime);
            
            allConcatenatedClassifierRes[classifierIndex] = concatenatedClassifierRes;
        }
   
        return allConcatenatedClassifierRes;
    }
    
    
//    public ClassifierResults[] stratifiedResampleWithStats(Classifier[] classifiers, Instances dataset) throws Exception {
//        if (cloneData)
//            dataset = new Instances(dataset);
//        if (cloneClassifiers)
//            cloneClassifiers(classifiers);
//        
//        resultsPerFold = new ClassifierResults[classifiers.length][numFolds];
//        
//        ClassifierResults[] allConcatenatedClassifierRes = new ClassifierResults[classifiers.length] ;
//        
//        for (int classifierIndex = 0; classifierIndex < classifiers.length; ++classifierIndex) {
//            
//            ClassifierResults concatenatedClassifierRes = new ClassifierResults(dataset.numClasses());
//            concatenatedClassifierRes.setTimeUnit(TimeUnit.NANOSECONDS);
//            concatenatedClassifierRes.turnOffZeroTimingsErrors();
//            
//            for (int fold = 0; fold < numFolds; fold++) {
//                Instances[] resampledData = InstanceTools.resampleInstances(dataset, seed, propInstancesInTrain);
//
//                Classifier foldClassifier = classifiers[classifierIndex];
//                if (cloneClassifiers)
//                    //use the clone instead
//                    foldClassifier = foldClassifiers[classifierIndex][fold];
//                
//                foldClassifier.buildClassifier(resampledData[0]);
//                ClassifierResults foldResults = new ClassifierResults(dataset.numClasses());
//                foldResults.setTimeUnit(TimeUnit.NANOSECONDS);
//                foldResults.turnOffZeroTimingsErrors();
//
//                //todo, implement this loop via SingleTestSetEvluator            
//                for (Instance testInst : resampledData[1]) {
//                    double classVal = testInst.classValue(); //save in case we're deleting next line
//                    if (setClassMissing)
//                        testInst.setClassMissing();
//
//                    long startTime = System.nanoTime();
//                    double[] dist = foldClassifier.distributionForInstance(testInst);
//                    long predTime = System.nanoTime()- startTime;
//
//                    foldResults.addPrediction(classVal, dist, indexOfMax(dist), predTime, "");
//                    concatenatedClassifierRes.addPrediction(classVal, dist, indexOfMax(dist), predTime, "");
//                }
//
//                foldResults.turnOnZeroTimingsErrors();
//                foldResults.findAllStatsOnce(); 
//                
//                resultsPerFold[classifierIndex][fold] = foldResults;
//            }
//            
//            concatenatedClassifierRes.turnOnZeroTimingsErrors();
//            allConcatenatedClassifierRes[classifierIndex] = concatenatedClassifierRes;
//        }
//   
//        return allConcatenatedClassifierRes;
//    }
}

