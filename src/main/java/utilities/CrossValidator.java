
package utilities;

import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;

/**
 * Start of a custom cross validation class, to be built on/optimised over time as
 * work with ensembles progresses
 * 
 * Initial push uses Jay's stratified folding code from HESCA
 * 
 * @author James 
 */
public class CrossValidator {
            
    private Integer seed = null;
    private int numFolds;
    private ArrayList<Instances> folds;
    private ArrayList<ArrayList<Integer>> foldIndexing;

    public CrossValidator() {
        this.seed = null;
        this.folds = null;
        this.foldIndexing = null;
        this.numFolds = 10;
    }

    public ArrayList<ArrayList<Integer>> getFoldIndices() { return foldIndexing; }
    
    public Integer getSeed() {
        return seed;
    }

    public void setSeed(Integer seed) {
        this.seed = seed;
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    /**
     * @return the index in the original train set of the instance found at folds.get(fold).get(indexInFold) 
     */
    public int getOriginalInstIndex(int fold, int indexInFold) {
        return foldIndexing.get(fold).get(indexInFold);
    }


    public ClassifierResults crossValidateWithStats(Classifier classifier, Instances train) throws Exception {
        return crossValidateWithStats(new Classifier[] { classifier }, train)[0];
    }
    
    /**
     * TODO return/report variance across folds too
     * 
     * Performs more extensive cross validation using dist for instance and 
     * returns more information. 
     * 
     * Each classifier is built/validated using the same subsets of the data provided 
     * i.e for each prediction, all classifiers will have trained on the exact same
     * subset data to have made that classification
     * 
     * If folds have already been defined (by a call to buildFolds()), will use those,
     * else will create them internally 
     * 
     * @return double[classifier][prediction]
     */
    public ClassifierResults[] crossValidateWithStats(Classifier[] classifiers, Instances train) throws Exception {
        long time=System.currentTimeMillis();
        if (folds == null)
            buildFolds(train);

        double[][] predictions = new double[classifiers.length][train.numInstances()];
        double[][][] distsForInsts = new double[classifiers.length][train.numInstances()][];
        double[][] foldaccs = new double[classifiers.length][numFolds];
        double[] classifierAccs = new double[classifiers.length];
        
        double pred;
        double[] dist;
        //for each fold as test
        for(int testFold = 0; testFold < numFolds; testFold++){
            Instances[] trainTest = buildTrainTestSet(testFold);

            //for each classifier in ensemble
            for (int c = 0; c < classifiers.length; ++c) {
                classifiers[c].buildClassifier(trainTest[0]);

                //for each test instance on this fold
                for(int i = 0; i < trainTest[1].numInstances(); i++){
                    int instIndex = getOriginalInstIndex(testFold, i);
                    
                    //classify and store prediction
                    dist = classifiers[c].distributionForInstance(trainTest[1].instance(i));
                    pred = indexOfMax(dist);
                    
                    distsForInsts[c][instIndex] = dist;
                    predictions[c][instIndex] = pred;
                    
                    if (pred == trainTest[1].instance(i).classValue()) {
                        ++foldaccs[c][testFold];
                        ++classifierAccs[c];
                    }
                }    
                
                foldaccs[c][testFold] /= trainTest[1].numInstances();
            }
        }
        
        //shove data into moduleresults objects 
        ClassifierResults[] results = new ClassifierResults[classifiers.length];
        double[] classVals = train.attributeToDoubleArray(train.classIndex());
        long t2=System.currentTimeMillis();
        for (int c = 0; c < classifiers.length; c++) {  
            classifierAccs[c] /= predictions[c].length;
            double stddevOverFolds = StatisticalUtilities.standardDeviation(foldaccs[c], false, classifierAccs[c]);
            results[c] = new ClassifierResults(classifierAccs[c], classVals, predictions[c], distsForInsts[c], stddevOverFolds, train.numClasses());
            results[c].buildTime=t2-time;
        }
        return results;
    }
    
    public double[] crossValidate(Classifier classifier, Instances train) throws Exception{
        return crossValidate(new Classifier[] { classifier }, train)[0];
    }

    /**
     * Performs simple crossvalidation (i.e only returns preds) on all classifiers provided 
     * using the same fold split for all
     * i.e for each prediction, all classifiers will have trained on the exact same
     * subset data to have made that classification
     * 
     * If folds have already been defined (by a call to buildFolds()), will use those,
     * else will create them internally 
     * 
     * @return double[classifier][prediction]
     */
    public double[][] crossValidate(Classifier[] classifiers, Instances train) throws Exception{
        if (folds == null)
            buildFolds(train);

        double pred;
        double[][] predictions = new double[classifiers.length][train.numInstances()];

        //for each fold as test
        for(int testFold = 0; testFold < numFolds; testFold++){
            Instances[] trainTest = buildTrainTestSet(testFold);

            //for each classifier in ensemble
            for (int c = 0; c < classifiers.length; ++c) {
                classifiers[c].buildClassifier(trainTest[0]);

                //for each test instance on this fold
                for(int i = 0; i < trainTest[1].numInstances(); i++){
                    //classify and store prediction
                    pred = classifiers[c].classifyInstance(trainTest[1].instance(i));
                    predictions[c][getOriginalInstIndex(testFold, i)] = pred;
                }    
            }
        }
        return predictions;
    }

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

//    public void buildFoldsAttmpt2(Instances train) throws Exception {
//        train = new Instances(train); //make copy
//        
//        Random r = null;
//        if(seed != null){
//            r = new Random(seed);
//        }else{
//            r = new Random();
//        }
//        
////        train.randomize(r); //only use of random is here
//        
//        folds = new ArrayList<Instances>();
//        foldIndexing = new ArrayList<ArrayList<Integer>>();
//
//        for(int i = 0; i < numFolds; i++){
//            folds.add(new Instances(train,0));
//            foldIndexing.add(new ArrayList<>());
//        }
//        
//        //calc class dists
//        int[] classCounts = new int[train.numClasses()];
//        double[] classDists = new double[train.numClasses()];
//        for (int i = 0; i < train.numInstances(); i++) 
//            classCounts[(int)train.get(i).classValue()]++;
//        for (int c = 0; c < train.numClasses(); c++) 
//            classDists[c] = (double)classCounts[c] / train.numInstances();
//        
//        //distribute insts into class groups, recording their original index
//        ArrayList<Instances> byClass = new ArrayList<>();
//        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
//        for(int i = 0; i < train.numClasses(); i++){
//            byClass.add(new Instances(train,0));
//            byClassIndices.add(new ArrayList<>());
//        }
//        
//        ArrayList<Integer> instanceIds = new ArrayList<>();
//        for(int i = 0; i < train.numInstances(); i++){
//            instanceIds.add(i);
//        }
//        Collections.shuffle(instanceIds, r);//only use of random is here
//        
//        for (int i = 0; i < instanceIds.size(); ++i) {
//            int instIndex = instanceIds.get(i);
//            int instClassVal = (int)train.instance(instIndex).classValue();
//            byClass.get(instClassVal).add(train.instance(instIndex));
//            byClassIndices.get(instClassVal).add(instIndex);
//        }
//        
//        int numInstsPerFold = train.numInstances() / numFolds;
//        int instsUsed = 0;
//        for(int fold = 0; fold < numFolds; fold++) { 
//            int numInstsToUse = numInstsPerFold;
//            if (fold < train.numInstances() % numFolds)       
//                numInstsToUse++;
//            
//            for (int c = 0; c < classDists.length; c++) {
//                if (fold == numFolds-1) { //last fold, use whatever's left
//                    while (!byClass.get(c).isEmpty()) {
//                        folds.get(fold).add(byClass.get(c).remove(0));
//                        foldIndexing.get(fold).add(byClassIndices.get(c).remove(0));
//                        
//                        instsUsed++;
//                    }
//                }
//                else {
//                    int numClassInstsToUse = (int)Math.round(classDists[c]*numInstsToUse); //returns as long ??....
//                    for (int i = 0; i < numClassInstsToUse; i++) {
//                        if (!byClass.get(c).isEmpty()) {
//                            folds.get(fold).add(byClass.get(c).remove(0));
//                            foldIndexing.get(fold).add(byClassIndices.get(c).remove(0));
//                            
//                            instsUsed++;
//                        }
//                    }
//                }
//            }
//        }
//        
//        if (instsUsed != train.numInstances())
//            throw new Exception("zipzopbippitybop");
//    }
    
    public void buildFolds(Instances train) throws Exception {
        train = new Instances(train); //make copy
        
        Random r = null;
        if(seed != null){
            r = new Random(seed);
        }else{
            r = new Random();
        }
        
        folds = new ArrayList<Instances>();
        foldIndexing = new ArrayList<ArrayList<Integer>>();

        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }
        
        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++)
            instanceIds.add(i);
        Collections.shuffle(instanceIds, r);//only use of random is here
        
        //distribute insts into class groups, recording their original index
        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }
        for (int i = 0; i < instanceIds.size(); ++i) {
            int instIndex = instanceIds.get(i);
            int instClassVal = (int)train.instance(instIndex).classValue();
            byClass.get(instClassVal).add(train.instance(instIndex));
            byClassIndices.get(instClassVal).add(instIndex);
        }
        
        //and get them back out, so now in class order but randomized within each each
        ArrayList<Integer> sortedByClassInstanceIds = new ArrayList<>();
        for (int c = 0; c < train.numClasses(); c++) 
            sortedByClassInstanceIds.addAll(byClassIndices.get(c));
        
        int start = 0;
        for(int fold = 0; fold < numFolds; fold++) { 
            int i = start;
            while (i < train.numInstances()) {
                folds.get(fold).add(train.instance(sortedByClassInstanceIds.get(i)));
                foldIndexing.get(fold).add(sortedByClassInstanceIds.get(i));
                i += numFolds;
            }
            start++;    
        }
        
    }
    
//    public void buildFoldsOldJay(Instances train) throws Exception {               
//        Random r = null;
//        if(seed != null){
//            r = new Random(seed);
//        }else{
//            r = new Random();
//        }
//
//        folds = new ArrayList<Instances>();
//        foldIndexing = new ArrayList<ArrayList<Integer>>();
//
//        for(int i = 0; i < numFolds; i++){
//            folds.add(new Instances(train,0));
//            foldIndexing.add(new ArrayList<>());
//        }
//
//        ArrayList<Integer> instanceIds = new ArrayList<>();
//        for(int i = 0; i < train.numInstances(); i++){
//            instanceIds.add(i);
//        }
//        Collections.shuffle(instanceIds, r);
//
//        ArrayList<Instances> byClass = new ArrayList<>();
//        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
//        for(int i = 0; i < train.numClasses(); i++){
//            byClass.add(new Instances(train,0));
//            byClassIndices.add(new ArrayList<>());
//        }
//
//        int thisInstanceId;
//        double thisClassVal;
//        for(int i = 0; i < train.numInstances(); i++){
//            thisInstanceId = instanceIds.get(i);
//            thisClassVal = train.instance(thisInstanceId).classValue();
//
//            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
//            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
//        }
//
//         // now stratify        
//        Instances strat = new Instances(train,0);
//        ArrayList<Integer> stratIndices = new ArrayList<>();
//        int stratCount = 0;
//        int[] classCounters = new int[train.numClasses()];
//
//        while(stratCount < train.numInstances()){
//
//            for(int c = 0; c < train.numClasses(); c++){
//                if(classCounters[c] < byClass.get(c).size()){
//                    strat.add(byClass.get(c).instance(classCounters[c]));
//                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
//                    classCounters[c]++;
//                    stratCount++;
//                }
//            }
//        }
//
//
//        train = strat;
//        instanceIds = stratIndices;
//
//        double foldSize = (double)train.numInstances()/numFolds;
//
//        double thisSum = 0;
//        double lastSum = 0;
//        int floor;
//        int foldSum = 0;
//
//
//        int currentStart = 0;
//        for(int f = 0; f < numFolds; f++){
//
//
//            thisSum = lastSum+foldSize+0.000000000001;  
//// to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
//            floor = (int)thisSum;
//
//            if(f==numFolds-1){
//                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
//            }
//
//            for(int i = currentStart; i < floor; i++){
//                folds.get(f).add(train.instance(i));
//                foldIndexing.get(f).add(instanceIds.get(i));
//            }
//
//            foldSum+=(floor-currentStart);
//            currentStart = floor;
//            lastSum = thisSum;
//        }
//
//        if(foldSum!=train.numInstances()){
//            throw new Exception("Error! Some instances got lost while creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
//        }
//    }
    
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
        buildFoldsTest(); 
//        CrossValidator cv = new CrossValidator();
//        cv.setNumFolds(10);
//        cv.setSeed(0);
//        
//        Classifier c = new kNN();
//        Instances insts = ClassifierTools.loadData("C:/TSC Problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
//        
//        double[] preds = cv.crossValidate(c, insts);
//        
//        double acc = 0.0;
//        System.out.println("Pred | Actual");
//        for (int i = 0; i < preds.length; i++) {
//            System.out.printf("%4d | %d\n", (int)preds[i], (int)insts.get(i).classValue());
//            if (preds[i] == insts.get(i).classValue())
//                ++acc;
//        }
//        
//        acc /= preds.length;
//        System.out.println("\n Acc: " + acc);
    }
    
    public static void buildFoldsTest() throws Exception {
        CrossValidator cv = new CrossValidator();
        cv.setNumFolds(3);
        cv.setSeed(0);
        
        String dset = "lenses";
//        String dset = "balloons";
//        String dset = "acute-inflammation";
        Instances insts = ClassifierTools.loadData("C:/UCI Problems/"+dset+"/"+dset);
        
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
    
    
}
