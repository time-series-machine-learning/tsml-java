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
import experiments.ClassifierLists;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import weka.classifiers.Classifier;
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
    
    @Override
    public Evaluator cloneEvaluator() {
        StratifiedResamplesEvaluator ev = new StratifiedResamplesEvaluator(this.seed, this.cloneData, this.setClassMissing, this.cloneClassifiers, this.maintainClassifiers);
        ev.setPropInstancesInTrain(this.propInstancesInTrain);
        ev.setUseEachResampleIdAsSeed(this.useEachResampleIdAsSeed);
        return ev;
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
    public synchronized ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        
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
    
    public synchronized ClassifierResults stratifiedResampleWithStats(Classifier classifier, Instances dataset) throws Exception {
        return stratifiedResampleWithStats(new Classifier[] { classifier }, dataset)[0];
    }
    
    public synchronized ClassifierResults[] stratifiedResampleWithStats(Classifier[] classifiers, Instances data) throws Exception {
        
        final Instances dataset = cloneData ? new Instances(data) : data;
       
        if (cloneClassifiers)
            cloneClassifiers(classifiers);
        
        resultsPerFold = new ClassifierResults[classifiers.length][numFolds];
        ClassifierResults[] allConcatenatedClassifierRes = new ClassifierResults[classifiers.length];
        
        //TODO obviously clean up this garbage once actual design is decided on 
        List<List<Future<ClassifierResults>>> futureResultsPerFold = new ArrayList<>(classifiers.length); //generic arrays... 
        for (int i = 0; i < classifiers.length; i++) {
            futureResultsPerFold.add(new ArrayList<>(numFolds));
            for (int j = 0; j < numFolds; j++)
                futureResultsPerFold.get(i).add(null);
        }
        if (multiThread)
            executor = Executors.newFixedThreadPool(numThreads);
        
        
        for (int classifierIndex = 0; classifierIndex < classifiers.length; ++classifierIndex) {
            
            //rebuild for each classifier so resamples are aligned
            //ignored if useEachResampleIdAsSeed == true
            Random classifierRng = new Random(seed); 
            
            long estimateTimeStart = System.nanoTime();
            
            for (int fold = 0; fold < numFolds; fold++) {
                final Classifier foldClassifier = cloneClassifiers ? foldClassifiers[classifierIndex][fold] : classifiers[classifierIndex];
                
                int resampleSeed = useEachResampleIdAsSeed ? fold : classifierRng.nextInt();
                String foldStr = "resample"+resampleSeed;    
                
                SingleSampleEvaluator eval = new SingleSampleEvaluator(resampleSeed, this.cloneData, this.setClassMissing);
                eval.setPropInstancesInTrain(this.propInstancesInTrain);
                
                Callable<ClassifierResults> foldEvalFunc = () -> {
                    long estimateTime = System.nanoTime();
                    ClassifierResults res = eval.evaluate(foldClassifier, dataset);
                    estimateTime = System.nanoTime() - estimateTime;
                    res.setErrorEstimateTime(estimateTime);
                    res.setDatasetName(res.getDatasetName()+"_"+foldStr);
                    return res;
                };
                
                if (!multiThread) {
                    //compute the result now
                    resultsPerFold[classifierIndex][fold] = foldEvalFunc.call();             
                    if (cloneClassifiers && !maintainClassifiers)
                        foldClassifiers[classifierIndex][fold] = null; //free the memory
                }
                else {
                    //spawn a job to compute the result, will collect it later
                    futureResultsPerFold.get(classifierIndex).set(fold, executor.submit(foldEvalFunc));
                }
            }
            
            if (multiThread) {
                //collect results from futures, this method will not continue until all folds done
                for (int fold = 0; fold < numFolds; fold++) {
                    resultsPerFold[classifierIndex][fold] = futureResultsPerFold.get(classifierIndex).get(fold).get();
                    if (cloneClassifiers && !maintainClassifiers)
                        foldClassifiers[classifierIndex][fold] = null; //free the memory
                }
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
            
        if (multiThread)
            executor.shutdown();
        
        return allConcatenatedClassifierRes;
    }
}

