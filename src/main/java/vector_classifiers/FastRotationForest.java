/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vector_classifiers;

import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierResults;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import vector_classifiers.TunedRandomForest;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.lazy.kNN;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Michael Flynn
 */
public class FastRotationForest extends RandomizableIteratedSingleClassifierEnhancer implements TrainAccuracyEstimate, SaveParameterInfo{
    private Random random;
    private int baggingProportion;
    private Instances trainingData;
    private ArrayList<Double> OOBErrors;
    //edce
    private ArrayList<Integer> testIndexs;
    private boolean weightBaseClassifiersByOOBError = false;
    private String trainPath = "";
    private boolean estimateAcc = false;
    private boolean[][] bagMatrix;
    private double[][][] classDistMatrix;
    private Instances bag;
    private double trainAcc;
    private ClassifierResults classifierResults;
    private boolean withReplacement;

    public FastRotationForest() {
        random = new Random(0);
        initialise();
    }

    private void initialise() {
        this.m_NumIterations = 200;
        baggingProportion = 70;
        withReplacement = true;
        this.m_Classifier = new RotationTree();
    }
    
    public void setWeightBaseClassifiersByOOBError(boolean x){
        weightBaseClassifiersByOOBError = x;
    }
    
    public void setBaggingProportion(int proportion){
        this.baggingProportion = proportion;
    }

    public void setWithReplacement(boolean withReplacement){
        this.withReplacement = withReplacement;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        //Make copy of data.
        trainingData = new Instances(data, 0, data.numInstances());
     
        super.buildClassifier(trainingData);
        OOBErrors = new ArrayList<>();
        bagMatrix = new boolean[m_NumIterations][trainingData.size()];
        classDistMatrix = new double[m_NumIterations][trainingData.size()][trainingData.numClasses()];
        trainAcc = 0.0;
        classifierResults = new ClassifierResults();
      
        long startTime = System.nanoTime();
        
        for (int i = 0; i < m_NumIterations; i++) {    
            bag = produceBag(trainingData, i);
            m_Classifiers[i].buildClassifier(bag);
            getOOBError(trainingData, i);     
            if (weightBaseClassifiersByOOBError)
                setWeights(m_Classifiers[i], trainingData);  
        }
        
        getTrainAcc(trainingData);
        classifierResults.buildTime = System.nanoTime() - startTime;
        generateTrainFiles(trainingData);
    }
    
    private void generateTrainFiles(Instances trainingData){
        if(!"".equals(trainPath)){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(trainingData.relationName()+", FastRotF, Train");
            f.writeLine(getParameters());
            f.writeLine(classifierResults.acc+"");
            f.writeLine(classifierResults.writeInstancePredictions());
        }
    }
    
    private void setWeights(Classifier classifier, Instances testBag) throws Exception{
        
        double oldSumOfWeights = 0;
        double newSumOfWeights = 0;
        
        oldSumOfWeights = testBag.sumOfWeights();
        
        //Classifier cCopy = AbstractClassifier.makeCopy(classifier);
        
        for (int i = 0; i < testBag.size(); i++) {
            if(classifier.classifyInstance(testBag.get(i)) != testBag.get(i).classValue()){
                //trainingData.get(i).setWeight(testBag.get(i).weight() * (classifier.distributionForInstance(testBag.get(i))[(int)testBag.get(i).classValue()]));
                trainingData.get(testIndexs.get(i)).setWeight(trainingData.get(i).weight() * (OOBErrors.get(OOBErrors.size()-1) * 10));
            }
        }
        
//        newSumOfWeights = trainingData.sumOfWeights();
//        
//        for (int i = 0; i < trainingData.size(); i++) {
//            trainingData.get(i).setWeight(trainingData.get(i).weight() * oldSumOfWeights / newSumOfWeights);
//        }
    }
    
    private Instances produceBag(Instances trainingData, int treeIndex){
        bag = new Instances(trainingData, 0);
        ArrayList<Integer> trainIndexs = new ArrayList<>();
        if (withReplacement) {
            withReplacement(treeIndex, trainIndexs);
        }else{
            withoutReplacement(treeIndex, trainIndexs);
        }
        testIndexs = new ArrayList<>();
        for (int i = 0; i < trainingData.size(); i++) {
            if(!containsIndex(trainIndexs, i)){
                testIndexs.add(i);
                bagMatrix[treeIndex][i] = false;
            }
        }
        return bag;
    }
    
    private void withReplacement(int treeIndex, ArrayList<Integer> trainIndexs){
        for (int i = 0; i < trainingData.size(); i++) {
            trainIndexs.add(random.nextInt(trainingData.size() - 0));
            bag.add(trainingData.get(trainIndexs.get(i)));
            bagMatrix[treeIndex][trainIndexs.get(i)] = true;
        }
    }
    
    private void withoutReplacement(int treeIndex, ArrayList<Integer> trainIndexs){
        int numInBag = (int)Math.floor(trainingData.size() * (baggingProportion / 100.0));
        for (int i = 0; i < numInBag; i++) {
            int index;
            do{
            index = random.nextInt(trainingData.size() - 0); 
            }while(containsIndex(trainIndexs, index));
            
            trainIndexs.add(index);
            bag.add(trainingData.get(trainIndexs.get(i)));
            bagMatrix[treeIndex][trainIndexs.get(i)] = true;
        } 
    }
//    private Instances[] produceBag(Instances trainingData){
//        Instances[] bag = new Instances[2];
//        bag[0] = new Instances(trainingData, 0, trainingData.size());
//        bag[0].clear();
//        bag[1] = new Instances(trainingData, 0, trainingData.size());
//        bag[1].clear();
//        
//        int numInBag = (int)Math.floor(trainingData.size() * (baggingProportion / 100.0));
//        int[] trainIndexs = new int[numInBag];
//        
//        double[] weights = new double[trainingData.size()];
//        
//        for (int i = 0; i < trainingData.size(); i++) {
//            weights[i] = trainingData.get(i).weight();
//        }
//        int [] sortedIndices = Utils.sort(weights);
//        
//        for (int i = 0; i < numInBag; i++) {
//            bag[0].add(trainingData.get(sortedIndices[(sortedIndices.length - i)-1]));
//            trainIndexs[i] = sortedIndices[(sortedIndices.length - i)-1];
//        }
//        testIndexs = new ArrayList<>();
//        for (int i = 0; i < trainingData.size(); i++) {
//            if(!containsIndex(trainIndexs, i)){
//                bag[1].add(trainingData.get(i));
//                testIndexs.add(i);
//            }
//        }
//        
//        return bag;
//    }
    
    private void getOOBError(Instances trainingData, int treeIndex){
        double error = 0.0;
        for (int i = 0; i < trainingData.size(); i++) {
            if (!bagMatrix[treeIndex][i]) {
                try {
                    classDistMatrix[treeIndex][i] = m_Classifiers[treeIndex].distributionForInstance(trainingData.get(i));
                    if (getMaxVoteFromDist(classDistMatrix[treeIndex][i]) != trainingData.get(i).classValue()) {
                        error++;
                    }
                } catch (Exception ex) {
                    System.out.println("This error has been thrown from getOOBError: treeIndex = " + treeIndex + ", trainindDataIndex = " + i);
                }
            }
        }
        OOBErrors.add(error/(trainingData.size() - bag.size()));
    }
    
    private boolean containsIndex(ArrayList<Integer> trainIndexs, int index){
        boolean flag = false;
        
        for (int i = 0; i < trainIndexs.size(); i++) {
            if(trainIndexs.get(i) == index)
                flag = true;
        }
        return flag;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[]distribution = distributionForInstance(instance);

        return getMaxVoteFromDist(distribution);
    }
    
    private int getMaxVoteFromDist(double[] distribution){
        int maxVote=0;
        for(int i = 1; i < distribution.length; i++)
            if(distribution[i] > distribution[maxVote])
                maxVote = i;
        return maxVote;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[]distribution = new double[instance.numClasses()];
        
        for (int i = 0; i < m_Classifiers.length; i++) {
            int classVal = 0;
            classVal = (int) m_Classifiers[i].classifyInstance(instance);
            if (weightBaseClassifiersByOOBError) {
                distribution[classVal] += OOBErrors.get(i);
            }else{
                distribution[classVal] ++;
            }
        }
        
        for(int i = 0 ; i < distribution.length; i++)
            distribution[i] /= m_Classifiers.length;
        
        return distribution;
    }
    
    private void getTrainAcc(Instances trainingData){
        double[] distribution;
        ArrayList<int[]> predAct = new ArrayList<>();
        
        for (int i = 0; i < trainingData.size(); i++) {
            distribution = new double[trainingData.numClasses()];
            for (int j = 0; j < bagMatrix.length; j++) {
                if (!bagMatrix[j][i]) {
                    for (int k = 0; k < distribution.length; k++) {
                        distribution[k] += this.classDistMatrix[j][i][k];
                    }
                }     
            }
            for (int j = 0; j < distribution.length; j++) {
                distribution[j] = distribution[j]/m_NumIterations;
            }
            predAct.add(new int[2]);
            predAct.get(predAct.size()-1)[0] = getMaxVoteFromDist(distribution);
            predAct.get(predAct.size()-1)[1] = (int)trainingData.get(i).classValue();
            classifierResults.storeSingleResult(trainingData.get(i).classValue(), distribution);
        }
        
        for (int i = 0; i < predAct.size(); i++) {
            if (predAct.get(i)[0] == predAct.get(i)[1]) {
                trainAcc++;
            }
        }
        classifierResults.acc = trainAcc/trainingData.size();
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean findsTrainAccuracyEstimate() {
        return TrainAccuracyEstimate.super.findsTrainAccuracyEstimate(); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
        estimateAcc=true;
    }
 @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        estimateAcc=setCV;
    }

    @Override
    public ClassifierResults getTrainResults() {
        return classifierResults;
    }

    @Override
    public int setNumberOfFolds(Instances data) {
        return TrainAccuracyEstimate.super.setNumberOfFolds(data); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String getParameters() {
        return  "BuildTime, " + classifierResults.buildTime + 
                ", OOBAcc, " + classifierResults.acc + 
                ", NumTrees, " + m_NumIterations + 
                ", withReplacement," + withReplacement + 
                ", baggingProportion (Only applicable if withReplacement = false)," + baggingProportion;
    }
    
}
