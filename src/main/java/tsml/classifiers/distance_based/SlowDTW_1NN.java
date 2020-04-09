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
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.ParameterSplittable;
import machine_learning.classifiers.SaveEachParameter;
import weka.core.*;

/* 
This classifier does the full 101 parameter searches for window. 
It is only here for comparison to faster methods
 */

public class SlowDTW_1NN extends EnhancedAbstractClassifier  implements SaveEachParameter,ParameterSplittable{
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
    
//Think this always does para search?
//    @Override
//    public boolean findsTrainAccuracyEstimate(){ return findTrainAcc;}


    public SlowDTW_1NN(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);    
        dtw=new DTW();
        accuracy=new ArrayList<>();
    }
    public SlowDTW_1NN(DTW_DistanceBasic d){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);    
        dtw=d;
        accuracy=new ArrayList<>();
    }
    @Override
    public String getParameters() {
        String result="BuildTime,"+trainResults.getBuildTime()+",CVAcc,"+trainResults.getAcc()+",Memory,"+trainResults.getMemory();
        result+=",BestWarpPercent,"+bestWarp+",AllAccs,";
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
            for(int i=maxPercentageWarp;i>=0;i-=1){
                dtw.setR(i/100.0);
            double acc=crossValidateAccuracy(maxAcc);
               
            accuracy.add(acc);
            if(acc>maxAcc){
                maxR=i;
                maxAcc=acc;
            }

            }
            bestWarp=(int)(maxR*dataLength/100);
            System.out.println("OPTIMAL WINDOW ="+maxR+" % which gives a warp of"+bestWarp+" data");
  //          dtw=new DTW();
            dtw.setR(maxR/100.0);
            trainResults.setAcc(maxAcc);
        }
        try {
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
            
            trainResults.setClassifierName("SlowDTW_1NN");
            trainResults.setDatasetName(train.relationName());
            trainResults.setSplit("train");
            //no foldid/seed
            trainResults.setNumClasses(train.numClasses());
            trainResults.setParas(getParameters());
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
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
        }
//        System.out.println("trainSize ="+trainSize+" stored ="+storeCount+" recovered "+recoverCount);
        return a/(double)trainSize;
    }
    public static void main(String[] args) throws Exception{
            SlowDTW_1NN c = new SlowDTW_1NN();
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
