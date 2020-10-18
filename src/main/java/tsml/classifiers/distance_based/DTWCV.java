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
package tsml.classifiers.distance_based;
import java.util.ArrayList;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW_DistanceBasic;
import java.util.HashMap;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.util.concurrent.TimeUnit;
import machine_learning.classifiers.SaveEachParameter;
import weka.core.*;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.ParameterSplittable;

/* 
 * The reason for specialising is this class has the option of searching for 
the optimal window length
 * through a grid search of values.
 * 
 * By default this class does not do a search of window size.  
 * To search for the window size call
 * optimiseWindow(true);
 * By default, this does a leave one out cross validation on every possible 
window size, then sets the proportion to the one with the largest accuracy,
ties taking the smallest window (slow)
. This will be slow, and not how the Keogh group do it. They do a stepwise increase
of window by 1% until there is no improvement for three steps. 

This has two possible speedups

1. Optimize window. This starts at full window, w=100%, and records the maximum warp 
 made over the data set, say k. Rather than move to w=w-1 it moves to w=k if k<w-1,
thus saving many evaluations

2. Early abandon on a window. If, during the accuracy calculation for a single window size,
the accuracy cannot be better than the best so far, we can quit. 

3. Early abandon on the nearest neighbour calculation. One obvious speed up is 
to store the distance matrix for a given window size. This requires O(n^2) extra
memory and means you cannot early abandon individual distances. 

O DO: 
DONE: avoid repeated evaluations for short series. Needs a debug
2. Set up check pointing


CHECK THIS: For implementation reasons, a window size of 1 
is equivalent to Euclidean distance (rather than a window size of 0
 */

public class DTWCV extends EnhancedAbstractClassifier implements SaveEachParameter,ParameterSplittable{
    private boolean optimiseWindow=false;
    private double windowSize=1;
    private int maxPercentageWarp=100;
    private Instances train;
    private int trainSize;
    private int bestWarp;
    private int maxWindowSize;
    DTW_DistanceBasic dtw;
    HashMap<Integer,Double> distances;
    double maxR=1;
    ArrayList<Double> accuracy=new ArrayList<>();
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
       
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }
    
    public void setFindTrainAccuracyEstimate(boolean setCV){
        if(setCV==true)
            throw new UnsupportedOperationException("Doing a top leve CV is not yet possible for DTWCV. It cross validates to optimize, so could store those, but will be biased"); //To change body of generated methods, choose Tools | Templates.
//This method doe
    }
     

    public DTWCV(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);        
        dtw=new DTW();
        accuracy=new ArrayList<>();
    }
    public DTWCV(DTW_DistanceBasic d){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);    
        dtw=d;
        accuracy=new ArrayList<>();
    }
    @Override
    public String getParameters() {
        String result="BuildTime,"+trainResults.getBuildTime()+",CVAcc,"+trainResults.getAcc()+",Memory,"+trainResults.getMemory();
        result+=",BestWarpPercent,"+bestWarp+"AllAccs,";
       for(double d:accuracy)
            result+=","+d;
        
        return result;
    }
     
   
    
    public double getMaxR(){ return maxR;}
    public void setMaxPercentageWarp(int a){maxPercentageWarp=a;}
    public void optimiseWindow(boolean b){ optimiseWindow=b;}
    public void setR(double r){dtw.setR(r);}
    public double getR(){ return dtw.getR();}
    public int getBestWarp(){ return bestWarp;}
    public int getWindowSize(){ return dtw.getWindowSize(train.numAttributes()-1);}

    @Override
    public void buildClassifier(Instances d) throws Exception{
        trainResults =new ClassifierResults();
        long t=System.nanoTime();
        
        train=d;
        trainSize=d.numInstances();
        if(optimiseWindow){
            maxR=0;
            double maxAcc=0;
            int dataLength=train.numAttributes()-1;
/*  If the data length < 100 then there will be some repetition
            should skip some values I reckon
            if(dataLength<maxNosWindows)
                maxPercentageWarp=dataLength;
        */
            double previousPercentage=0;
            for(int i=maxPercentageWarp;i>=0;i-=1){
        //Set r for current value as the precentage of series length.
//                dtw=new DTW();
               
                int previousWindowSize=dtw.findWindowSize(previousPercentage,d.numAttributes()-1);
                int newWindowSize=dtw.findWindowSize(i/100.0,d.numAttributes()-1);
                if(previousWindowSize==newWindowSize)// no point doing this one
                    continue;
                previousWindowSize=newWindowSize;
                dtw.setR(i/100.0);
                        
                        
/*Can do an early abandon inside cross validate. If it cannot be more accurate 
 than maxR even with some left to evaluate then stop evaluation
*/                
                double acc=crossValidateAccuracy(maxAcc);
                accuracy.add(acc);
                if(acc>maxAcc){
                    maxR=i;
                    maxAcc=acc;
               }
//                System.out.println(" r="+i+" warpsize ="+x+" train acc= "+acc+" best acc ="+maxR);
/* Can ignore all window sizes bigger than the max used on the previous iteration
*/                
                
               if(maxWindowSize<(i-1)*dataLength/100){
                   System.out.println("WINDOW SIZE ="+dtw.getWindowSize()+" Can reset downwards at "+i+"% to ="+((int)(100*(maxWindowSize/(double)dataLength))));
                   i=(int)(100*(maxWindowSize/(double)dataLength));
                   i++;
//                   i=Math.round(100*(maxWindowSize/(double)dataLength))/100;
               } 

            }
            bestWarp=(int)(maxR*dataLength/100);
            System.out.println("OPTIMAL WINDOW ="+maxR+" % which gives a warp of"+bestWarp+" data");
  //          dtw=new DTW();
            dtw.setR(maxR/100.0);
            trainResults.setAcc(maxAcc);
        }
        try {
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(System.nanoTime()-t);
        } catch (Exception e) {
            System.err.println("Inheritance preventing me from throwing this error...");
            System.err.println(e);
        }
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        trainResults.setMemory(usedBytes);
        
        if(getEstimateOwnPerformance()){  //Save basic train results
            long estTime = System.nanoTime();
            for(int i=0;i<train.numInstances();i++){
                Instance test=train.remove(i);
                
                long predTime = System.nanoTime();
                int pred=(int)classifyInstance(test);
                predTime = System.nanoTime() - predTime;
                
                double[] dist = new double[train.numClasses()];
                dist[pred] = 1.0;
                
                trainResults.addPrediction(test.classValue(), dist, pred, predTime, "");
                    
                train.add(i,test);
            }
            estTime = System.nanoTime() - estTime;
            trainResults.setErrorEstimateTime(estTime);
            trainResults.setErrorEstimateMethod("cv_loo");
            
            trainResults.setClassifierName("DTWCV");
            trainResults.setDatasetName(train.relationName());
            trainResults.setSplit("train");
            //no foldid/seed
            trainResults.setNumClasses(train.numClasses());
            trainResults.setParas(getParameters());
            trainResults.finaliseResults();
        }        
        
    }
    @Override
    public double classifyInstance(Instance d){
/*Basic distance, with early abandon. This is only for 1-nearest neighbour*/
            double minSoFar=Double.MAX_VALUE;
            double dist; int index=0;
            for(int i=0;i<train.numInstances();i++){
                    dist=dtw.distance(train.instance(i),d,minSoFar);
                    if(dist<minSoFar){
                            minSoFar=dist;
                            index=i;
                    }
            }
            return train.instance(index).classValue();
    }
    @Override
    public double[] distributionForInstance(Instance instance){
        double[] dist=new double[instance.numClasses()];
        dist[(int)classifyInstance(instance)]=1;
        return dist;
    }

    
    /**Could do this by calculating the distance matrix, but then 	
 * you cannot use the early abandon. Early abandon about doubles the speed,
 * as will storing the distances. Given the extra n^2 memory, probably better
 * to just use the early abandon. We could store those that were not abandoned?
answer is to store those without the abandon in a hash table indexed by i and j,
*where index i,j == j,i

* @return 
 */
    private  double crossValidateAccuracy(double maxAcc){
        double a=0,d, minDist;
        int nearest;
        Instance inst;
        int bestNosCorrect=(int)(maxAcc*trainSize);
        maxWindowSize=0;
        int w;
        distances=new HashMap<>(trainSize);
        
        
        for(int i=0;i<trainSize;i++){
//Find nearest to element i
            nearest=0;
            minDist=Double.MAX_VALUE;
            inst=train.instance(i);
            for(int j=0;j<trainSize;j++){
                if(i!=j){
//  d=dtw.distance(inst,train.instance(j),minDist);
//Store past distances if not early abandoned 
//Not seen i,j before                    
                  if(j>i){
                        d=dtw.distance(inst,train.instance(j),minDist);
                        //Store if not early abandon
                        if(d!=Double.MAX_VALUE){
//                            System.out.println(" Storing distance "+i+" "+j+" d="+d+" with key "+(i*trainSize+j));
                            distances.put(i*trainSize+j,d);
//                            storeCount++;
                        }
//Else if stored recover                        
                    }else if(distances.containsKey(j*trainSize+i)){
                        d=distances.get(j*trainSize+i);
//                       System.out.println(" Recovering distance "+i+" "+j+" d="+d);
//                        recoverCount++;
                    }
//Else recalculate with new early abandon                    
                    else{
                        d=dtw.distance(inst,train.instance(j),minDist);
                    }        
                    if(d<minDist){
                        nearest=j;
                        minDist=d;
                        w=dtw.findMaxWindow();
                        if(w>maxWindowSize)
                            maxWindowSize=w;
                    }
                }
            }
                //Measure accuracy for nearest to element i			
            if(inst.classValue()==train.instance(nearest).classValue())
                a++;
           //Early abandon if it cannot be better than the best so far. 
            if(a+trainSize-i<bestNosCorrect){
//                    System.out.println(" Early abandon on CV when a="+a+" and i ="+i+" best nos correct = "+bestNosCorrect+" maxAcc ="+maxAcc+" train set size ="+trainSize);
                return 0.0;
            }
        }
//        System.out.println("trainSize ="+trainSize+" stored ="+storeCount+" recovered "+recoverCount);
        return a/(double)trainSize;
    }
    public static void main(String[] args) throws Exception{
            DTWCV c = new DTWCV();
            String path="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

            Instances test=DatasetLoading.loadDataNullable(path+"Coffee\\Coffee_TEST.arff");
            Instances train=DatasetLoading.loadDataNullable(path+"Coffee\\Coffee_TRAIN.arff");
            train.setClassIndex(train.numAttributes()-1);
            c.buildClassifier(train);

    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setParamSearch(boolean b) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setParametersFromIndex(int x) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }


}
