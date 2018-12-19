package timeseriesweka.classifiers;
import fileIO.OutFile;
import java.util.ArrayList;
import timeseriesweka.elastic_distance_measures.DTW;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import java.util.HashMap;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import vector_classifiers.SaveEachParameter;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

/* 
This classifier does the full 101 parameter searches for window. 
It is only here for comparison to faster methods
 */

public class SlowDTW_1NN extends AbstractClassifier  implements SaveParameterInfo, TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable{
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
    String trainPath;
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
    private ClassifierResults res =new ClassifierResults();
    
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }
 @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
    }  
    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        if(setCV==true)
            throw new UnsupportedOperationException("Doing a top leve CV is not yet possible for SlowDTW_1NN. It cross validates to optimize, so could store those, but will be biased"); //To change body of generated methods, choose Tools | Templates.
//This method doe
    }
    
    
//Think this always does para search?
//    @Override
//    public boolean findsTrainAccuracyEstimate(){ return findTrainAcc;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
        return res;
    }      
      @Override
    public String getParas() { //This is redundant really.
        return getParameters();
    }

    @Override
    public double getAcc() {
        return res.acc;
    }  

    public SlowDTW_1NN(){
        dtw=new DTW();
        accuracy=new ArrayList<>();
    }
    public SlowDTW_1NN(DTW_DistanceBasic d){
        dtw=d;
        accuracy=new ArrayList<>();
    }
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",Memory,"+res.memory;
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
    public void buildClassifier(Instances d){
        res =new ClassifierResults();
        long t=System.currentTimeMillis();
        
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
            res.acc=maxAcc;
        }
        res.buildTime=System.currentTimeMillis()-t;
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        res.memory=usedBytes;
        
        
        if(trainPath!=null && trainPath!=""){  //Save basic train results
//            
//NEED TO FIND THE TRAIN ESTIMATES FOR EACH TEST HERE            
            OutFile f= new OutFile(trainPath);
            f.writeLine(train.relationName()+",FastDTW_1NN,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            for(int i=0;i<train.numInstances();i++){
                Instance test=train.remove(i);
                int pred=(int)classifyInstance(test);
                f.writeString((int)test.classValue()+","+pred+",");
                for(int j=0;j<train.numClasses();j++){
                    if(j==pred)
                        f.writeString(",1");
                    else
                        f.writeString(",0");
                }
                f.writeString("\n");
                train.add(i,test);
            }
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
    public static void main(String[] args){
            SlowDTW_1NN c = new SlowDTW_1NN();
            String path="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

            Instances test=ClassifierTools.loadData(path+"Coffee\\Coffee_TEST.arff");
            Instances train=ClassifierTools.loadData(path+"Coffee\\Coffee_TRAIN.arff");
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
