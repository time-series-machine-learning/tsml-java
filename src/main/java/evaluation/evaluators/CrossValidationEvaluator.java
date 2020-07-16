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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * An evaluator that performs k-fold crossvalidation (default k=10) on the given s
 * data and evaluates the given classifier(s) on each fold. 
 * 
 * Concatenated predictions across all folds are returned from the main 
 * evaluate method, however predictions split across each fold can also be retrieved
 * afterwards
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CrossValidationEvaluator extends MultiSamplingEvaluator {
      
    private String previousRelationName = "EmPtY";
    
    private ArrayList<Instances> folds;
    private ArrayList<ArrayList<Integer>> foldIndexing;

    public CrossValidationEvaluator() {
        super(0,false,false,false,false);
        
        this.folds = null;
        this.foldIndexing = null;
        this.numFolds = 10;
    }

    public CrossValidationEvaluator(int numFolds) {
        this();
        setNumFolds(numFolds);
    }
    
    public CrossValidationEvaluator(int seed, boolean cloneData, boolean setClassMissing, boolean cloneClassifiers, boolean maintainClassifiers) {
        super(seed,cloneData,setClassMissing, cloneClassifiers, maintainClassifiers);
        
        this.folds = null;
        this.foldIndexing = null;
        this.numFolds = 10;
    }

    public ArrayList<ArrayList<Integer>> getFoldIndices() { return foldIndexing; }

    /**
     * @return the index in the original train set of the instance found at folds.get(fold).get(indexInFold) 
     */
    public int getOriginalInstIndex(int fold, int indexInFold) {
        return foldIndexing.get(fold).get(indexInFold);
    }

    private void checkNumCVFolds(int numInstances) { 
        if (numInstances < numFolds)
            numFolds = numInstances;
    }

    @Override
    public synchronized ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        ClassifierResults res = crossValidateWithStats(classifier, dataset);
        res.findAllStatsOnce();
        return res;
    }
    
    public synchronized ClassifierResults crossValidateWithStats(Classifier classifier, Instances dataset) throws Exception {
        return crossValidateWithStats(new Classifier[] { classifier }, dataset)[0];
    }
    
    /**
     * Performs more extensive cross validation using dist for instance and 
     * returns more information. 
     * 
     * Each classifier is built/validated using the same subsets of the data provided 
     * i.e for each prediction, all classifiers will have trained on the exact same
     * subset data to have made that classification
     * 
     * If folds have already been defined (by a call to buildFolds()), will use those,
     * else will create them internally. Setting the seed makes folds reproducable
     * across different instantiations of this object
     * 
     * @return double[classifier][prediction]
     */
    public synchronized ClassifierResults[] crossValidateWithStats(Classifier[] classifiers, final Instances dataset) throws Exception {
        
        if (folds == null || !previousRelationName.equals(dataset.relationName()))
            buildFolds(dataset);
        
        if (cloneClassifiers)
            cloneClassifiers(classifiers);
        
        //store for later storage of results, in case we want to set the class values missing
        //on each instance at predict time
        double[] trueClassVals = dataset.attributeToDoubleArray(dataset.classIndex());
        
        resultsPerFold = new ClassifierResults[classifiers.length][numFolds];
        
        //TODO obviously clean up this garbage once actual design is decided on 
        List<List<Future<ClassifierResults>>> futureResultsPerFold = new ArrayList<>(classifiers.length); //generic arrays... 
        for (int i = 0; i < classifiers.length; i++) {
            futureResultsPerFold.add(new ArrayList<>(numFolds));
            for (int j = 0; j < numFolds; j++)
                futureResultsPerFold.get(i).add(null);
        }
        if (multiThread)
            executor = Executors.newFixedThreadPool(numThreads);
        
        //for each fold as test
        for(int fold = 0; fold < numFolds; fold++){
            Instances[] trainTest = buildTrainTestSet(fold);
            final Instances train = trainTest[0];
            final Instances test = trainTest[1];
            
            String foldStr = "cvFold"+fold;
            
            //for each classifier in ensemble
            for (int classifierIndex = 0; classifierIndex < classifiers.length; ++classifierIndex) {
                
                // get the classifier instance to be used this fold
                final Classifier foldClassifier = cloneClassifiers ? foldClassifiers[classifierIndex][fold] : classifiers[classifierIndex];
                final SingleTestSetEvaluator tester = new SingleTestSetEvaluator(seed, cloneData, setClassMissing);
                
                Callable<ClassifierResults> eval = () -> {
                    long estimateTime = System.nanoTime();
                    ClassifierResults res = tester.evaluate(foldClassifier, train, test);
                    estimateTime = System.nanoTime() - estimateTime;
                    res.setErrorEstimateTime(estimateTime);
                    res.setDatasetName(res.getDatasetName()+"_"+foldStr);
                    return res;
                };
                
                if (!multiThread) {
                    //compute the result now
                    resultsPerFold[classifierIndex][fold] = eval.call();                    
                    if (cloneClassifiers && !maintainClassifiers)
                        foldClassifiers[classifierIndex][fold] = null; //free the memory
                }
                else {
                    futureResultsPerFold.get(classifierIndex).set(fold, executor.submit(eval));
                }
            }
        }
        
        if (multiThread) {
            //collect results from futures, this method will not continue until all folds done
            for (int fold = 0; fold < numFolds; fold++) {
                for (int classifierIndex = 0; classifierIndex < classifiers.length; ++classifierIndex) {
                    resultsPerFold[classifierIndex][fold] = futureResultsPerFold.get(classifierIndex).get(fold).get();
                    if (cloneClassifiers && !maintainClassifiers)
                        foldClassifiers[classifierIndex][fold] = null; //free the memory
                }
            }
            executor.shutdown();
        }
        
        
        //shove concatenated fold data into ClassifierResults objects, the singular form
        //to represent the entire cv process (trainFoldX)
        //and get predictions for instances as ordered in original train set, instead of 
        //the order predicted in 
        //todo maybe implement flag to turn this off/on, bespoke to cv really
        ClassifierResults[] results = new ClassifierResults[classifiers.length];
        for (int c = 0; c < classifiers.length; c++) {
            results[c] = concatenateAndReorderFoldPredictions(resultsPerFold[c], 
                    classifiers[c].getClass().getSimpleName(), 
                    dataset.relationName(), 
                    trueClassVals);
        }

        return results;
    }
    
    private ClassifierResults concatenateAndReorderFoldPredictions(ClassifierResults[] foldResults, String fullClassifierName, String fullDatasetName, double[] trueClassVals) throws Exception {
        ClassifierResults res = new ClassifierResults(foldResults[0].numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        res.setClassifierName(fullClassifierName);
        res.setDatasetName(fullDatasetName);
        res.setFoldID(seed);
        res.setSplit("train"); //todo revisit, or leave with the assumption that calling method will set this to test when needed

        res.turnOffZeroTimingsErrors();

        double[][] dists = new double[trueClassVals.length][];
        double[] preds = new double[trueClassVals.length];
        long[] times = new long[trueClassVals.length];
        String[] descs = new String[trueClassVals.length];

        long totalBuildTime = 0;
        long totalEstimateTime = 0;

        for (int fold = 0; fold < numFolds; fold++) {
            String foldStr = "cvFold"+fold;

            //has the preds in order predicted for this fold
            ClassifierResults foldRes = foldResults[fold];
            totalBuildTime += foldRes.getBuildTime();
            totalEstimateTime += foldRes.getErrorEstimateTime();

            for (int i = 0; i < foldRes.numInstances(); i++) {
                //get them out as original order in train set
                int originalIndex = getOriginalInstIndex(fold, i);

                double[] dist = foldRes.getProbabilityDistribution(i);
                dists[originalIndex] = dist;
                times[originalIndex] = foldRes.getPredictionTime(i);
                descs[originalIndex] = foldStr+foldRes.getPredDescription(i);

                //crossvalidator always resolved ties randomly, continued for reproducability
                //even if the lower-level evaluator resolved ties e.g. naively per fold
                //todo review
                double tiesResolvedRandomlyPred;
                tiesResolvedRandomlyPred = indexOfMax(dist);

                preds[originalIndex] = tiesResolvedRandomlyPred;
            }
        }

        res.addAllPredictions(trueClassVals, preds, dists, times, descs);
        res.setBuildTime(totalBuildTime);
        res.turnOnZeroTimingsErrors();

        //have put the total build time before errors being turned back on,
        //e.g. ED1NN might legitimately get 0 build time for each fold, but for 
        //all classifiers at least a FEW predictions should take more than ~200 
        //nanoseconds
        res.setErrorEstimateTime(totalEstimateTime);
        
        return res;
    }
    
//    public synchronized ClassifierResults[] crossValidateWithStats(Classifier[] classifiers, final Instances dataset) throws Exception {
//        
//        if (folds == null || !previousRelationName.equals(dataset.relationName()))
//            buildFolds(dataset);
//        
//        if (cloneClassifiers)
//            cloneClassifiers(classifiers);
//        
//        //store for later storage of results, in case we want to set the class values missing
//        //on each instance at predict time
//        double[] trueClassVals = dataset.attributeToDoubleArray(dataset.classIndex());
//        
//        //these will store dists and preds for instance AS THEY ARE ORDERED IN THE DATASET GIVEN
//        //as opposed to instances in the order that they are predicted, after having been split into the k folds.
//        //storing them here in order, then adding into the classifierresults objects in order after the actual 
//        //cv has finished
//        double[][][] allFolds_distsForInsts = new double[classifiers.length][dataset.numInstances()][];
//        long[][] allFolds_predTimes = new long[classifiers.length][dataset.numInstances()];
//        long[] totalEstimateTimes = new long[classifiers.length];
//        
//        resultsPerFold = new ClassifierResults[classifiers.length][numFolds];
//        
//        //for each fold as test
//        for(int fold = 0; fold < numFolds; fold++){
//            Instances[] trainTest = buildTrainTestSet(fold);
//            final Instances train = trainTest[0];
//            final Instances test = trainTest[1];
//
//            //for each classifier in ensemble
//            for (int classifierIndex = 0; classifierIndex < classifiers.length; ++classifierIndex) {
//                
//                // get the classifier instance to be used this fold
//                Classifier foldClassifier = classifiers[classifierIndex];
//                if (cloneClassifiers)
//                    //use the clone instead
//                    foldClassifier = foldClassifiers[classifierIndex][fold];
//               
//                long foldEstimateTimeStart = System.nanoTime(); //for errorEstimateTime of the full results object
//                long foldBuildTime = foldEstimateTimeStart;         //for the buildtime of this fold's results object 
//                foldClassifier.buildClassifier(train);
//                foldBuildTime = System.nanoTime() - foldBuildTime;
//                
//                // init the classifierXfold results object
//                ClassifierResults classifierFoldRes = new ClassifierResults(dataset.numClasses());
//                classifierFoldRes.setTimeUnit(TimeUnit.NANOSECONDS);
//                classifierFoldRes.setClassifierName(foldClassifier.getClass().getSimpleName());
//                classifierFoldRes.setDatasetName(dataset.relationName()+"_cvfold"+fold);
//                classifierFoldRes.setFoldID(seed);
//                classifierFoldRes.setSplit("train"); 
//                classifierFoldRes.turnOffZeroTimingsErrors();
//                classifierFoldRes.setBuildTime(foldBuildTime);
//
//                //for each test instance on this fold
//                for(int i = 0; i < test.numInstances(); i++){
//                    int instIndex = getOriginalInstIndex(fold, i);
//                    
//                    Instance testInst = test.instance(i);
//                    
//                    double classVal = testInst.classValue(); //save in case we're deleting next line
//                    if (setClassMissing)
//                        testInst.setClassMissing();
//                    
//                    //classify and store prediction
//                    long startTime = System.nanoTime();
//                    double[] dist = foldClassifier.distributionForInstance(testInst);
//                    long predTime = System.nanoTime()- startTime;
//                    
//                    allFolds_distsForInsts[classifierIndex][instIndex] = dist;
//                    allFolds_predTimes[classifierIndex][instIndex] = predTime;
//
//                    classifierFoldRes.addPrediction(classVal, dist, indexOfMax(dist), predTime, "");
//                }    
//                
//                long foldEstimateTime = System.nanoTime() - foldEstimateTimeStart;
//                totalEstimateTimes[classifierIndex] += foldEstimateTime;
//                
//                classifierFoldRes.turnOnZeroTimingsErrors();
//                classifierFoldRes.finaliseResults();
//                classifierFoldRes.findAllStatsOnce();
//                resultsPerFold[classifierIndex][fold] = classifierFoldRes;
//                
//                if (cloneClassifiers && !maintainClassifiers)
//                    foldClassifiers[classifierIndex][fold] = null; //free the memory
//            }
//        }
//        
//        //shove concatenated fold data into ClassifierResults objects, the singular form
//        //to represent the entire cv process (trainFoldX)
//        ClassifierResults[] results = new ClassifierResults[classifiers.length];
//        for (int c = 0; c < classifiers.length; c++) {
//            results[c] = new ClassifierResults(dataset.numClasses());
//            results[c].setTimeUnit(TimeUnit.NANOSECONDS);
//            results[c].setClassifierName(classifiers[c].getClass().getSimpleName());
//            results[c].setDatasetName(dataset.relationName());
//            results[c].setFoldID(seed);
//            results[c].setSplit("train"); //todo revisit, or leave with the assumption that calling method will set this to test when needed
//            
//            results[c].turnOffZeroTimingsErrors();
//            results[c].setErrorEstimateTime(totalEstimateTimes[c]); 
//            for (int i = 0; i < dataset.numInstances(); i++) {
//                double tiesResolvedRandomlyPred;
//
//                tiesResolvedRandomlyPred = indexOfMax(allFolds_distsForInsts[c][i]);
//
//                results[c].addPrediction(allFolds_distsForInsts[c][i], tiesResolvedRandomlyPred, allFolds_predTimes[c][i], "");
//            }
//            results[c].turnOnZeroTimingsErrors();
//            
//            results[c].finaliseResults(trueClassVals);
//        }
//
//        return results;
//    }
    

    /**
     * @return [0] = new train set, [1] = test(validation) set
     */
    public Instances[] buildTrainTestSet(int testFold) {
        Instances[] trainTest = new Instances[2];
        trainTest[0] = null;
        trainTest[1] = new Instances(folds.get(testFold));

        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake
        for(int f = 0; f < folds.size(); f++){
            if(f==testFold){
                continue;
            }
            temp = new Instances(folds.get(f));
            if(trainTest[0]==null){
                trainTest[0] = temp;
            }else{
                trainTest[0].addAll(temp);
            }
        }

        return trainTest;
    }

    public void buildFolds(Instances dataset) throws Exception {
        previousRelationName = dataset.relationName();
        
        if (cloneData)
            dataset = new Instances(dataset); //make copy
        
        checkNumCVFolds(dataset.numInstances());
        Random r = new Random(seed);
        
        folds = new ArrayList<Instances>();
        foldIndexing = new ArrayList<ArrayList<Integer>>();

        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(dataset,0));
            foldIndexing.add(new ArrayList<>());
        }
        
        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < dataset.numInstances(); i++)
            instanceIds.add(i);
        Collections.shuffle(instanceIds, r);//only use of random is here
        
        //distribute insts into class groups, recording their original index
        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < dataset.numClasses(); i++){
            byClass.add(new Instances(dataset,0));
            byClassIndices.add(new ArrayList<>());
        }
        for (int i = 0; i < instanceIds.size(); ++i) {
            int instIndex = instanceIds.get(i);
            int instClassVal;

            instClassVal = (int)dataset.instance(instIndex).classValue();

            byClass.get(instClassVal).add(dataset.instance(instIndex));
            byClassIndices.get(instClassVal).add(instIndex);
        }
        
        //and get them back out, so now in class order but randomized within each each
        ArrayList<Integer> sortedByClassInstanceIds = new ArrayList<>();
        for (int c = 0; c < dataset.numClasses(); c++) 
            sortedByClassInstanceIds.addAll(byClassIndices.get(c));
        
        int start = 0;
        for(int fold = 0; fold < numFolds; fold++) { 
            int i = start;
            while (i < dataset.numInstances()) {
                folds.get(fold).add(dataset.instance(sortedByClassInstanceIds.get(i)));
                foldIndexing.get(fold).add(sortedByClassInstanceIds.get(i));
                i += numFolds;
            }
            start++;    
        }
        
    }
    
    private double indexOfMax(double[] dist) {
        double  bsfWeight = -(Double.MAX_VALUE);
        ArrayList<Integer>  bsfClassVals = null;
        
        for (int c = 0; c < dist.length; c++) {
            if(dist[c] > bsfWeight){
                bsfWeight = dist[c];
                bsfClassVals = new ArrayList<>();
                bsfClassVals.add(c);
            }else if(dist[c] == bsfWeight){
                bsfClassVals.add(c);
            }
        }
        double pred; 
        //if there's a tie for highest voted class after all modules have voted, settle randomly
        if(bsfClassVals.size()>1)
            pred = bsfClassVals.get(new Random(0).nextInt(bsfClassVals.size()));
        else
            pred = bsfClassVals.get(0);
        
        return pred;
    }
    
    
    public static void main(String[] args) throws Exception {
//        buildFoldsTest(); 
        classifierCloningTest();
    }
    
    public static void classifierCloningTest() throws Exception { 
        String resLoc = "C:/Temp/crossvalidatortests/";
        String dataLoc = "C:/TSC Problems/";
        
        String dset = "ItalyPowerDemand";
        String[] classifierNames = { "MLP", "SVML", "Logistic", "C45", "NN" };
        int numResamples = 5;
            
        for (String classifierName : classifierNames) {
            System.out.println(classifierName);
            for (int resample = 0; resample < numResamples; resample++) {
                Instances[] data = DatasetLoading.sampleDataset(dataLoc, dset, resample);
                Classifier classifier = ClassifierLists.setClassifierClassic(classifierName, resample);
                
                CrossValidationEvaluator cv = new CrossValidationEvaluator(resample, true, false, true, true);
                ClassifierResults fullcvResults = cv.evaluate(classifier, data[0]);
                System.out.println("\tdataset resample "+resample+" cv acc: "+fullcvResults.getAcc());
                
                for (int fold = 0; fold < cv.numFolds; fold++) {
                    ClassifierResults foldClassifierResultsOnValFold = cv.resultsPerFold[0][fold];
                    System.out.println("\t\t cv fold "+fold+": "+foldClassifierResultsOnValFold.getAcc());
                    
                    
                    SingleTestSetEvaluator testeval = new SingleTestSetEvaluator(resample, true, false);
                    ClassifierResults foldClassifierResultsOnFullTest = testeval.evaluate(cv.foldClassifiers[0][fold], data[1]);
                    System.out.println("\t\t fold "+fold+" classiifer on test: "+foldClassifierResultsOnFullTest.getAcc());
                }
                
                classifier.buildClassifier(data[0]);
                SingleTestSetEvaluator testeval = new SingleTestSetEvaluator(resample, true, false);
                System.out.println("\tfull train set test acc : " + testeval.evaluate(classifier, data[1]).getAcc());
                
            }
            System.out.println("");
        }
    }
    
    public static void buildFoldsTest() throws Exception {
        CrossValidationEvaluator cv = new CrossValidationEvaluator();
        cv.setNumFolds(3);
        cv.setSeed(0);
        
        String dset = "lenses";
//        String dset = "balloons";
//        String dset = "acute-inflammation";
        Instances insts = DatasetLoading.loadDataNullable("C:/UCI Problems/"+dset+"/"+dset);
        
        System.out.println("Full data:");
        System.out.println("numinsts="+insts.numInstances());
        
        int[] classCounts = new int[insts.numClasses()];
        double[] classDists = new double[insts.numClasses()];
        for (int j = 0; j < insts.numInstances(); j++) 
            classCounts[(int)insts.get(j).classValue()]++;
        for (int j = 0; j < insts.numClasses(); j++) 
            classDists[j] = (double)classCounts[j] / insts.numInstances();
        System.out.println("classcounts= " +Arrays.toString(classCounts));
        System.out.println("classdist=   " +Arrays.toString(classDists));
        
        
        cv.buildFolds(insts);
        for (int i = 0; i < cv.numFolds; i++) {
            Instances fold = cv.folds.get(i);
            
            System.out.println("\nFold " + i);
            System.out.println("numinsts="+fold.numInstances());
            
            int[] classCount = new int[insts.numClasses()];
            double[] classDist = new double[fold.numClasses()];
            for (int j = 0; j < fold.numInstances(); j++) 
                classCount[(int)fold.get(j).classValue()]++;
            for (int j = 0; j < fold.numClasses(); j++) 
                classDist[j] = (double)classCount[j] / fold.numInstances();
            System.out.println("classcounts= " +Arrays.toString(classCount));
            System.out.println("classdist=   " +Arrays.toString(classDist));
            
            
            Collections.sort(cv.foldIndexing.get(i));
            System.out.println("(sorted) orginal indices: " + cv.foldIndexing.get(i));
//            for (int j = 0; j < fold.numInstances(); j++) 
//                System.out.print(cv.foldIndexing.get(i).get(j)+",");
            System.out.println("");
        }
        
    }

    @Override
    public Evaluator cloneEvaluator() {
        CrossValidationEvaluator ev = new CrossValidationEvaluator(this.seed, this.cloneData, this.setClassMissing, this.cloneClassifiers, this.maintainClassifiers);
        //INTENTIONALLY NOT COPYING ACROSS FOLDS. That is a utility to help speed things up
        
        //If people try to clone evaluators with folds already built, safer to force
        //folds to be rebuilt (seeded/deterministic, ofc) than to potentially create
        //many copies of large datasets
        return ev;
    }
    
    
}
