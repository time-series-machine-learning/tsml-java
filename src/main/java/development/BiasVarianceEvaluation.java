/*
Methods to perform a biasKohavi/variance decomposition. Note there is a class
BVDecompose.java in weka



1. Perform multiple train/test samples, record the test prediction on each instance

instance 1, trueClass, predFold1, .... predFoldk1
instance 2, trueClass, predFold1, .... predFoldk2

instances will be in different test folds and may be in test a varying number of times.

Suppose we have c=3, k=10, true class =1, testPredictions

1,1,2,2,1,1,1,1,3,3

generates 0/1 loss vector over possible classes

x_1=1,x_2=0,x_3=0

then we have a probability of each class for the instance
p_1=3/5, p_2=1/5, p_3=1/5

Then sum over all instances the biasKohavi
sum_i=1^c (x_i-p_i)^2 -  p_i * (1 - p_i) / (k-1);

and the varianceKohavi
1-sum_i=1^c p_i^2.

Bias estimate is 
for (int j = 0; j < numClasses; j++) {
        pActual = (current.classValue() == j) ? 1 : 0; // Or via 1NN from test data?
        pPred = predProbs[j] / m_TrainIterations;
        bsum += (pActual - pPred) * (pActual - pPred)
          - pPred * (1 - pPred) / (m_TrainIterations - 1);

    double [][] instanceProbs = new double [numTest][numClasses];



    m_Error = 0;
//1. COUNT HOW MANY PREDICTED FOR EACH CLASS
    for (int i = 0; i < m_TrainIterations; i++) {
      if (m_Debug) {
        System.err.println("Iteration " + (i + 1));
      }
      trainPool.randomize(random);
      Instances train = new Instances(trainPool, 0, m_TrainPoolSize / 2);

      Classifier current = AbstractClassifier.makeCopy(m_Classifier);
      current.buildClassifier(train);

      //// Evaluate the classifier on test, updating BVD stats
      for (int j = 0; j < numTest; j++) {
        int pred = (int)current.classifyInstance(test.instance(j));
        if (pred != test.instance(j).classValue()) {
          m_Error++;
        }
        instanceProbs[j][pred]++;
      }
    }
//Output p_j = proportion of classifications for each class

//2. FIND BIAS and VARIANCE FOR EACH INSTANCE
    // Average the BV over each instance in test.
    m_Bias = 0;
    m_Variance = 0;
    m_Sigma = 0;
    for (int i = 0; i < numTest; i++) {
      Instance current = test.instance(i);
      double [] predProbs = instanceProbs[i];
      double pActual, pPred;
      double bsum = 0, vsum = 0, ssum = 0;
//FOR EACH CLASS j=1 to c, biasKohavi = sum [ (x_j -p_j)^2 -p_j*(1-p_j)/(k-1) 
      for (int j = 0; j < numClasses; j++) {
        pActual = (current.classValue() == j) ? 1 : 0; // Or via 1NN from test data?
        pPred = predProbs[j] / m_TrainIterations;
        bsum += (pActual - pPred) * (pActual - pPred)
          - pPred * (1 - pPred) / (m_TrainIterations - 1);
        vsum += pPred * pPred;
        ssum += pActual * pActual;
      }
      m_Bias += bsum;
      m_Variance += (1 - vsum);
      m_Sigma += (1 - ssum);
    }
    m_Bias /= (2 * numTest);
    m_Variance /= (2 * numTest);
    m_Sigma /= (2 * numTest);

 */
package development;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.TreeSet;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class BiasVarianceEvaluation {
    int numTest;
    int numClasses;
    double bias;
    double variance;
/*
 The test data labels are in allActual   
 allPreds is the train testPredictions, allPreds[i][j] is the prediction of the 
    jth resample on the ith instance
    */
    public static class BVResult{
/**
 * There are several ways of doing this. The Kohavi way (kohavi96bv: 
 * "Bias Plus Variance Decomposition for Zero-One Loss Functions" follows the 
 * Weka version in BVDecompose. The Domingos version follows 
 * domingos00bv "A unified .." and is I think generally preferred. valentini00bv_svm
 * ("Bias-variance analysis of support vector machines ...") is a good background 
 
 */
        public double biasKohavi;
        public double varianceKohavi;
        public double biasDomingos;
        public double varianceDomingos;
        public double unbiasedVariance;
        public double biasedVariance;
        public double netVariance; //
        public double acc=0;
        BVResult(){
            biasKohavi=0;
            varianceKohavi=0;
        }
        @Override
        public String toString(){
            return acc+",,"+biasKohavi+","+varianceKohavi+",,"+biasDomingos+","+varianceDomingos+","+unbiasedVariance+","+biasedVariance+","+netVariance;
        }
    }
    public static BVResult findBV(String algo, String problem){
        InFile f=new InFile(resultsPath+"BV/"+algo+"/"+problem+"TEST.csv");
        int lines=f.countLines();
        f=new InFile(resultsPath+"BV/"+algo+"/"+problem+"TEST.csv");
        int[] actual=new int[lines];
        int[][] preds=new int[lines][];
        
        
        for(int i=0;i<lines;i++){
            String[] line=f.readLine().split(",");
//            System.out.println("Number in line "+i+" ="+line.length);
            actual[i]=Integer.parseInt(line[1]);
            int p=line.length-3;
            preds[i]=new int[p];
            for(int j=0;j<p;j++)
                preds[i][j]=Integer.parseInt(line[j+3]);
        }
        TreeSet<Integer> ts=new TreeSet<>();
        for(int i:actual)
            ts.add(i);
        int numClasses=ts.size();
        System.out.println("Number of classes ="+numClasses);
        BVResult bv=findBV(actual,preds,numClasses);
        System.out.println(algo+ "BV for "+problem+" = "+bv.toString());
        return bv;
    } 
    
/*    public static BVResult findBVDomingos(int[] allActual,int[][] allPreds, int numClasses){
        int numCases=allActual.length;
        BVResult res=new BVResult();
            System.out.println("num instances  ="+numCases);
        int correct=0;
        int count=0;
        for(int i=0;i<numCases;i++){
            int[] preds=allPreds[i];
            int actual=allActual[i];
            double[] p = new double[numClasses];
            int numClassifiers=preds.length;
//            System.out.println("Instance "+i+" has "+numClassifiers+" predictions and "+numClasses+" classes");
            for(int j:preds){
                p[j]++;
                count++;
                if(j==actual)
                    correct++;
            }
            double bSum=0,vSum=0;
            for (int j = 0; j < numClasses; j++) {
              int x = (actual == j) ? 1 : 0; 
              bSum += (x - p[j])*(x - p[j])- p[j] * (1 - p[j]) / (numClassifiers - 1);
              vSum += p[j]*p[j];
            }
            res.biasKohavi+=bSum;
            res.varianceKohavi+=(1-vSum);
//            System.out.println(": biasKohavi ="+bSum+" Variance ="+(1-vSum));
        }
        res.acc=(double)correct/(double)count;
        res.biasKohavi/=numCases;
        res.varianceKohavi/=numCases;
        System.out.println("BIAS ="+res.biasKohavi+" VARIANCE = "+res.varianceKohavi);
        return res;
    } 
*/    
    public static BVResult findBV(int[] allActual,int[][] allPreds, int numClasses){
        int numCases=allActual.length;
        BVResult res=new BVResult();
            System.out.println("num instances  ="+numCases);
        int correct=0;
        int count=0;
        int[] mainPredictions=new int[numCases];
        for(int i=0;i<numCases;i++){
            int[] preds=allPreds[i];
            int actual=allActual[i];
            double[] p = new double[numClasses];
            int numClassifiers=preds.length;
            
            //if this case only has 0 or 1 predictions for it, will break the analysis
            //when calcing bSum += (x - p[j])*(x - p[j])- p[j] * (1 - p[j]) / (numClassifiers - 1);
            //so just skip this case
            if (numClassifiers < 2)
                continue;
            
//            System.out.println("Instance "+i+" has "+numClassifiers+" predictions and "+numClasses+" classes");
            for(int j:preds){
                p[j]++;
                count++;
                if(j==actual)
                    correct++;
            }
//Find main prediction y_m. Domingos uses this for bias rather than probability.            
            mainPredictions[i]=0;
            for(int j=0;j<p.length;j++){
                if(p[j]>p[mainPredictions[i]])
                    mainPredictions[i]=j;
            }
            for(int j=0;j<p.length;j++)
                p[j]/=numClassifiers;
//Kohavi Bias and variance: probability weighted            
            double bSum=0,vSum=0;
            for (int j = 0; j < numClasses; j++) {
              int x = (actual == j) ? 1 : 0; 
              bSum += (x - p[j])*(x - p[j])- p[j] * (1 - p[j]) / (numClassifiers - 1);
              vSum += p[j]*p[j];
            }
            res.biasKohavi+=bSum;
            res.varianceKohavi+=(1-vSum);
//Domingos bias: 0/1 based on mode of predictions y_m (Eq 8 of valentini00            
            if(mainPredictions[i]!=actual)
                res.biasDomingos++;
//Domingos variance: mean over all predictions of the deviation from the bias
            double v=0;
            for(int j=0;j<preds.length;j++){
                if(preds[j]!=mainPredictions[i])
                    v++;
            }
            res.varianceDomingos+=v/preds.length;
// Unbiased and biased variance: split dependent on whether y_m=actual              
            if(mainPredictions[i]==actual){ //Correct: undesirable unbiased variance
                res.unbiasedVariance+=v/preds.length;
                res.netVariance+=v/preds.length;
            }
            else{   //Wrong: in this case, variance is desirable
                res.biasedVariance+=v/preds.length;
                res.netVariance-=v/preds.length;
            }
        }
        res.acc=(double)correct/(double)count;
        res.biasKohavi/=numCases;
        res.varianceKohavi/=numCases;
        res.biasDomingos/=numCases;
        res.varianceDomingos/=numCases;
        res.biasedVariance/=numCases;
        res.unbiasedVariance/=numCases;
        res.netVariance/=numCases;
        System.out.println("K_BIAS ="+res.biasKohavi+" K_VARIANCE = "+res.varianceKohavi);
        System.out.println("D_BIAS ="+res.biasDomingos+" D_VARIANCE = "+res.varianceDomingos+" UNBIASED_VAR="+res.unbiasedVariance+" BIASED_VAR="+res.biasedVariance+" NET_VAR="+res.netVariance);
        return res;
    } 

/**
 * 
 * @param problem
 * @param folds 
 * 
 * This method inputs
 *      1: The train and test arrfs in <dataPath><problem>/<problem>_Train.arff
      2. The testPredictions file that describes where each instance in the train/test 
 fold appear in the resample. This should be in <resultsPath>"Folds/"<problem>".csv"
      3. The test testPredictions for each fold. This should be in 
              <resultsPath><algorithm>"/Predictions/"<problem>tesFold<i>.csv
 
 1. Get the true class values in the order they appear in the original data
  store in array actualClassVals. Train fold in pos 0...trainSize-1, test in trainSize...numInstances-1
 
 2. Get all the resample fold mappings from the Folds directory. 
 int[][] resampleMap
 
 resampleMap[i][j]
 * resampleMap[i][j] gives the mapping FROM original place TO the place in fold i
 * SO resampleMap[1][2] =5, this means that the 3rd element is in position 6
 * in the second fold
 * 
 * 
  testPredictions[i] is the ArrayList for fold i
  testPredictions[i].get(j) will give the position in the resample i of element j in actualClassValues[j] once ca

3. Recover each testFold set of testPredictions, recover which element was in each test fold, then store the predicted values
*/    
    
    public static String resultsPath="C:/Data/BVTest/";
    public static String dataPath="C:/Data/BVTest/";
    public static String algorithm="RIF_PS_ACF";
    public static String[] fileNames=DataSets.tscProblems85;
/**
 * 
 * @param problem 
 * 
 * 1. Load test and train
 * 2. Add an attribute for an index
 * 3. Do the train test split
 * 4. Record indices
 */    
    
    public static void findMappings(String problem){
        Instances train=ClassifierTools.loadData(dataPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(dataPath+problem+"/"+problem+"_TEST");
        train.insertAttributeAt(new Attribute("Index"), 0);
        test.insertAttributeAt(new Attribute("Index"), 0);
        int value=0;
        for(Instance in:train){
            in.setValue(0, value++);
        }
        for(Instance in:test){
            in.setValue(0, value++);
        }
        OutFile folds=new OutFile(resultsPath+"Folds/"+problem+".csv");
        for(int i=0;i<100;i++){
            Instances[] split = InstanceTools.resampleTrainAndTestInstances(train, test, i);
            folds.writeLine("Fold,"+i);
            int count=0;
            for(Instance in:split[0]){
                if(in.value(0)<train.numInstances())
                    folds.writeLine(count+","+(int)in.value(0)+",TRAIN,TRAIN");
                else
                    folds.writeLine(count+","+(int)in.value(0)+",TEST,TRAIN");
                count++;
            }
            for(Instance in:split[1]){
                if(in.value(0)<train.numInstances())
                    folds.writeLine(count+","+(int)in.value(0)+",TRAIN,TEST");
                else
                    folds.writeLine(count+","+(int)in.value(0)+",TEST,TEST");
                count++;
            }
        }
    }    
    
    public static void formatPredictions(String problem, int folds){
        File f=new File(resultsPath+"Folds/"+problem+".csv");
        File f2=new File(resultsPath+algorithm+"/Predictions");
        if(!f.exists()){
            System.out.println("ERROR, file "+f+" not present");
            System.exit(0);
        }
        if(!f2.exists()){
            System.out.println("ERROR, file "+f2+" not present");
            System.exit(0);
        }
//Need to load up testPredictions for each test fold then map them back to the correct original
//        String dataSet=DataSets.dropboxPath+"TSC Problems/"+problem;
        String dataSet=resultsPath+problem;
        
        Instances train=ClassifierTools.loadData(dataPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(dataPath+problem+"/"+problem+"_TEST");
        int numInstances=train.numInstances()+test.numInstances();
        int trainSize=train.numInstances();
        int testSize=test.numInstances();
        System.out.println(" NUMBER OF INSTANCES="+numInstances);
        System.out.println("Train size ="+trainSize+" test size ="+testSize);
                int [] actualClassVals=new int[numInstances];
        int pos=0;
        for(Instance t:train){
            actualClassVals[pos++]=(int)t.classValue();
//            System.out.println(" Pos ="+(pos-1)+" class val ="+actualClassVals[pos-1]);
        }
        for(Instance t:test){
            actualClassVals[pos++]=(int)t.classValue();
//            System.out.println(" Pos ="+(pos-1)+" class val ="+actualClassVals[pos-1]);
        }
        InFile allFolds=new InFile(resultsPath+"Folds/"+problem+".csv");
        int[][] resampleMap=new int[folds][numInstances];
        
        ArrayList<Integer>[] trainPredictions;
        ArrayList<Integer>[] testPredictions;
        testPredictions = new ArrayList[numInstances];
        trainPredictions = new ArrayList[numInstances];
        for(int j=0;j<numInstances;j++){
            testPredictions[j]=new ArrayList<>();
            trainPredictions[j]=new ArrayList<>();
        }
        
        
        
        for(int i=0;i<folds;i++){
//Load mapping            
           String str2=allFolds.readLine();
//            System.out.println("Starting "+str2);
           for(int j=0;j<numInstances;j++){
// Original index, not needed, but check it should equal j
               int a=allFolds.readInt();
               if(a!=j){
                   System.out.println("ERROR in reading in the mapping, misalligned somehow. DEBUG");
                   System.exit(0);
               }
//Train/Test, not needed
               int map=allFolds.readInt();
//               System.out.println(j+","+a+","+map);
               resampleMap[i][j]=map;
               allFolds.readString();//Either TRAIN or TEST in the original split, not needed
               allFolds.readString();//Either TRAIN or TEST in the new split, not needed
           }
//Load up test fold testPredictions
//            InFile trainPredF=new InFile(resultsPath+algorithm+"/Predictions/"+problem+"/trainFold"+i+".csv");
            InFile testPredF=new InFile(resultsPath+algorithm+"/Predictions/"+problem+"/testFold"+i+".csv");
            for(int k=0;k<3;k++){ //Remove unnecessary header info 
                testPredF.readLine();
//                trainPredF.readLine();
            }
//           for(int j=0;j<numInstances;j++)
//               System.out.println("MAP: "+j+"  -> "+resampleMap[i][j]);
            
//Read in the actual and the predicted train and test into a single array 
//However, what if the train set has been subsampled? these files will now be smaller
//And we should not use these results, so maybe just ignore completelyÂ¬!             
            int[][] temp=new int[2][numInstances];
//            for(int j=0;j<trainSize;j++){
//                temp[0][j]=trainPredF.readInt();
//                temp[1][j]=trainPredF.readInt();
//                String s=trainPredF.readLine();
//                if(i==1)
//                System.out.println(" TRAIN Inst: "+temp[0][j]+", "+temp[1][j]);
//            }
            for(int j=trainSize;j<numInstances;j++){
                temp[0][j]=testPredF.readInt();
                temp[1][j]=testPredF.readInt();
                String s=testPredF.readLine();
//                if(i==1)
//                System.out.println(j+" "+s+" TEST Inst: "+temp[0][j]+", "+temp[1][j]);
            }
//Split these into the arrays according to the mapping            

           for(int j=0;j<numInstances;j++){
//Element j
               int foldPos=resampleMap[i][j]; //This is the position in temp of this instance
//                  System.out.println(" Element "+j+" in position "+foldPos+" on fold "+i);
               int actual=temp[0][j];
               int predicted=temp[1][j];
               if(j>=trainSize && actual!=actualClassVals[foldPos]){
               System.out.println(problem+" CLASS MISSMATCH Resample pos ="+j+"  Original position ="+foldPos);
                   System.out.println("CLASS MISMATCH IN ACTUAL..... NEEDS DEBUGGING");
                   System.out.println("Instance resample pos ="+j+" in test fold +"+i+" pos ="+(j-trainSize)+" position in original ="+foldPos+" Class from array ="+actualClassVals[foldPos]+" class from TestFold class ="+actual+" instance ="+(j));
                   Instance x;
                   if(foldPos<trainSize){
                      System.out.println(" Instance in train ");
                      x=train.instance(foldPos);
                   }
                   else{
                      System.out.println(" Instance in test first value");                       
                      x=test.instance(foldPos-trainSize);
                   }
                   System.out.println("class ="+x.classValue()+" first val="+x.value(0));

                   System.exit(0);
               }
               if(j<trainSize){    //In the train fold
//Check actual verses stored and original file for sanity sake
                   trainPredictions[foldPos].add(predicted);
               }
               else{     //In test fold
                   testPredictions[foldPos].add(predicted);
               }
           }
           //allFolds.readLine();
//            System.out.println("END FOLD "+i);
        }
//This seems correct ...write it all to file
        OutFile testF=new OutFile(resultsPath+"BV/"+algorithm+"/"+problem+"Test.csv");
        for(int i=0;i<testPredictions.length;i++){
            testF.writeString(i+","+actualClassVals[i]+",");
            for(Integer in: testPredictions[i])
                testF.writeString(","+in);
            testF.writeString("\n");
        }
    }
    public static void fullBV(String algo){
        algorithm=algo;
        String problem;
        OutFile bv=new OutFile(resultsPath+"BV/"+algorithm+"BV.csv");
        for(int i=59;i<fileNames.length;i++)
        {
            problem=fileNames[i];
            System.out.println(i+"PROBLEM FILE :"+problem);
//            findMappings(problem);
            formatPredictions(problem,100);
            
            BVResult b=findBV(algorithm,problem);
            bv.writeLine(problem+","+b.toString());
        }
//        findBV();
//        String s="SanityCheck";
//        formatPredictions(s,4);
//Generate test arff, test preds for 2 folds and test mappings
        
    }
    public static void main(String[] args) {
        dataPath="C:/Users/ajb/Dropbox/TSC Problems/";
        resultsPath="C:/Users/ajb/Dropbox/NewCOTEResults/";
//        dataPath="/gpfs/home/ajb/TSC Problems/";
//        resultsPath="/gpfs/home/ajb/Results/";
        
//        fullBV("BOSS");
//        fullBV("TSF");
//       fullBV("RIF_PS_ACF");
        fullBV("RotF");
        
        
        
    }
    
}
