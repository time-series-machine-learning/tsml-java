/*
learning rate of 0 meaningless, but 1 fine.  Momentum between 0 and 1 (inclusive) 
is fine.  Decay is probably weight decay regularisation,  difficult to know 
how it is coded and value will depend on the size of the training set and learning rate, I'd try 1, 1/2, 1/4, ... going down to very small values.
MLP parameters searched by this classifier

Two hiddn layer version:
number of hidden layers: 2 values, 1 or 2
number of nodes per layer: 4 values, data dependent, same for both if two hidden layers: 
Embedded in m_hiddenLayers. Values either 
"a"=(m_numAttributes + m_numClasses) / 2;
"i"= m_numAttributes;
"o" = m_numClasses;
"t" = m_numAttributes + m_numClasses;
Learning rate: 8 values, data independent, variable m_learningRate
1,1/2,1/4,1/2^7
Momentum: 8 values
0,1/8,2/8,7/8
m_decay 2 values: either true or false

total models = 4*2*2*8*8=1024

 */
package vector_classifiers;

import development.CollateResults;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.meta.RotationForest;
import utilities.ClassifierResults;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class TunedTwoLayerMLP extends MultilayerPerceptron implements SaveParameterInfo,TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable{
    protected boolean tuneParameters=true;
    protected String[] paraSpace1;//number of layers and number of nodes
    protected double[] paraSpace2;//LearningRate
    protected double[] paraSpace3;//Momentum
    protected boolean[] paraSpace4;//Decay: true or false, do this to keep loops sane
    
    protected String trainPath="";
    protected boolean debug=false;
    protected boolean findTrainAcc=true;
    protected int seed; //need this to seed cver/the forests for consistency in meta-classification/ensembling purposes
    protected Random rng; //legacy, 'seed' still (and always has) seeds this for any other rng purposes, e.g tie resolution
    protected ArrayList<Double> accuracy;
    protected ArrayList<Double> buildTimes;
    protected ClassifierResults res =new ClassifierResults();
    protected long combinedBuildTime;
    protected static int MAX_FOLDS=10;
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
    private static int MAX_PER_PARA=10;
    
    public TunedTwoLayerMLP(){
        super();
        rng=new Random();
        seed=0;
        accuracy=new ArrayList<>();
        setHiddenLayers("a,a");
        
        
    }   
//SaveParameterInfo    
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",Nodes,"+this.getHiddenLayers()+",LearningRate,"+this.getLearningRate()+",Momentum,"+this.getMomentum()+",Decay,"+this.getDecay();
        for(double d:accuracy)
            result+=","+d;       
        return result;
    }

    @Override
    public void setParamSearch(boolean b) {
        tuneParameters=b;
    }
 //methods from SaveEachParameter    
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }   
//MEthods from ParameterSplittable    
    @Override
    public String getParas() { //This is redundant really.
        return getParameters();
    }
    @Override
    public double getAcc() {
        return res.acc;
    }
    @Override
    public void setParametersFromIndex(int x) {
        x=x-1;
//Para 1: structure: 8 settings
//Para 2: Learning rate: 8 settings
//Para 3: Momentum: 8 settings
//Para 4: Decay: 2 settings
//        int paraSpaceSize=8*8*8*2=1024;
        int para1Value=x/128; // First block para 1
        int remainder=x%128;
        int para2Value=remainder/16;
        remainder=remainder%16;
        int para3Value=remainder/2;
        int para4Value=x%2;
        System.out.println(para1Value+","+para2Value+","+para3Value+","+para4Value);
        switch(para1Value){
            case 0:
                this.setHiddenLayers("a");
            break;
            case 1:
                this.setHiddenLayers("i");
            break;
            case 2:
                this.setHiddenLayers("o");
            break;
            case 3:
                this.setHiddenLayers("t");
            break;
            case 4:
                this.setHiddenLayers("a,a");
            break;
            case 5:
                this.setHiddenLayers("i,i");
            break;
            case 6:
                this.setHiddenLayers("o,o");
            break;
            case 7:
                this.setHiddenLayers("t,t");
            break;
        }
        this.setLearningRate(1.0/Math.pow(2,para2Value));

 // SET UP MODEL        
        this.setMomentum(((double)para3Value)/10.0);
        if(para4Value==0)
            this.setDecay(false);
        else
            this.setDecay(true);
    }
    public void setSeed(int s){
        super.setSeed(s);
        seed = s;
        rng=new Random();
        rng.setSeed(seed);
    }
     public void debug(boolean b){
        this.debug=b;
    }
     public void justBuildTheClassifier(){
        estimateAccFromTrain(false);
        tuneParameters(false);
        debug=false;
    }

     public void estimateAccFromTrain(boolean b){
        this.findTrainAcc=b;
    }
  
    public void tuneParameters(boolean b){
        tuneParameters=b;
    }
 @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
        findTrainAcc=true;
    }    
 @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        findTrainAcc=setCV;
    }
    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return findTrainAcc;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
//        res.acc=ensembleCvAcc;
//TO DO: Write the other stats        
        return res;
    }        
    public final void setStandardParaSearchSpace(int m){
        paraSpace1=new String[8]; //# hidden layers
        paraSpace1[0]="a";// =(m_numAttributes + m_numClasses) / 2;
        paraSpace1[1]="i";//=m_numAttributes;
        paraSpace1[2]="o";// = m_numClasses;
        paraSpace1[3]="t";//= m_numAttributes + m_numClasses;    
        paraSpace1[4]="a,a";// =(m_numAttributes + m_numClasses) / 2;
        paraSpace1[5]="i,i";//=m_numAttributes;
        paraSpace1[6]="o,o";// = m_numClasses;
        paraSpace1[7]="t,t";//= m_numAttributes + m_numClasses;    
        paraSpace2=new double[8];//Learning rate
        for(int i=0;i<paraSpace2.length;i++)
            paraSpace2[i]=1.0/Math.pow(2,i);
        paraSpace3=new double[8];//momentum
        for(int i=0;i<paraSpace3.length;i++)
            paraSpace3[i]=i/10.0;
        paraSpace4=new boolean[2];
        paraSpace4[0]=true;
        paraSpace4[1]=false;
        System.out.println("Number of parameters for each ="+paraSpace1.length*paraSpace2.length*paraSpace3.length*paraSpace4.length);
            
    }

    public void tuneMLP(Instances train) throws Exception {
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();
        double minErr=1;
        this.setSeed(rng.nextInt());
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(String p1:paraSpace1){//Number of hidden nodes
            for(double p2:paraSpace2){//learning rate
                for(double p3:paraSpace3){//Num trees
                    for(boolean p4:paraSpace4){
                        count++;
    //                    System.out.println("Count ="+count);
                        if(saveEachParaAcc){// check if para value already done
                            File f=new File(resultsPath+count+".csv");
                            if(f.exists()){
    //                            System.out.println(resultsPath+count+".csv EXISTS");
                                if(CollateResults.validateSingleFoldFile(resultsPath+count+".csv")==false){
                                    System.out.println("Deleted file "+resultsPath+count+".csv because size ="+f.length());
                                }
                                else{
    //                                System.out.println("SKIPPING THIS FOLD");
                                    continue;//If done, ignore skip this iteration                        
                                }
                            }
                        }
                        TunedTwoLayerMLP model = new TunedTwoLayerMLP();
                        model.tuneParameters(false);
                        model.findTrainAcc=false;
                        model.setHiddenLayers(p1);
                        model.setLearningRate(p2);
                        model.setMomentum(p3);
                        model.setDecay(p4);

                        tempResults=cv.crossValidateWithStats(model,trainCopy);
                        tempResults.setName("MLPPara"+count);
                        tempResults.setParas("HiddenNodes,"+p1+",LearningRate,"+p2+",Momentum,"+p3+",Decay"+p4);
                        double e=1-tempResults.acc;
                        if(debug)
                            System.out.println("HiddenNodes,"+p1+",LearningRate,"+p2+",Momentum,"+p3+",Decay"+p4+", Acc = "+(1-e));
                        accuracy.add(tempResults.acc);
                        if(saveEachParaAcc){// Save to file and close
                            temp=new OutFile(resultsPath+count+".csv");
                            temp.writeLine(tempResults.writeResultsFileToString());
                            temp.closeFile();
                        }                
                        else{
                            if(e<minErr){
                            minErr=e;
                            ties=new ArrayList<>();//Remove previous ties
                            ties.add(new ResultsHolder(p1,p2,p3,p4,tempResults));
                            }
                            else if(e==minErr){//Sort out ties
                                ties.add(new ResultsHolder(p1,p2,p3,p4,tempResults));
                            }
                        }
                    }
                }
            }
        }
        String bestNumNodes;
        double bestLearningRate;
        double bestMomentum;
        boolean bestDecay;
        minErr=1;
        if(saveEachParaAcc){
// Check they are all there first. 
            int missing=0;
            for(String p1:paraSpace1){
                for(double p2:paraSpace2){
                    for(double p3:paraSpace3){
                        for(boolean p4:paraSpace4){
                            File f=new File(resultsPath+count+".csv");
                            if(!(f.exists() && f.length()>0))
                                missing++;
                        }
                    }
                }
            }
            if(missing==0)//All present
            {
                combinedBuildTime=0;
    //            If so, read them all from file, pick the best
                count=0;
                for(String p1:paraSpace1){
                    for(double p2:paraSpace2){
                        for(double p3:paraSpace3){
                            for(boolean p4:paraSpace4){
                                count++;
                                tempResults = new ClassifierResults();
                                tempResults.loadFromFile(resultsPath+count+".csv");
                                combinedBuildTime+=tempResults.buildTime;
                                double e=1-tempResults.acc;
                                if(e<minErr){
                                    minErr=e;
                                    ties=new ArrayList<>();//Remove previous ties
                                    ties.add(new ResultsHolder(p1,p2,p3,p4,tempResults));
                                }
                                else if(e==minErr){//Sort out ties
                                        ties.add(new ResultsHolder(p1,p2,p3,p4,tempResults));
                                }
                            }
                        }            
                    }
                }
                ResultsHolder best=ties.get(rng.nextInt(ties.size()));
                bestNumNodes=best.nodes;
                bestLearningRate=best.lRate;
                bestMomentum=best.mRate;
                bestDecay=best.decay;
/* SET UP FINAL MODEL */
                this.setHiddenLayers(bestNumNodes);
                this.setLearningRate(bestLearningRate);
                this.setMomentum(bestMomentum);
                this.setDecay(bestDecay);
                res=best.res;
//File clean up, delete all the individual fold files
        //Delete the files here to clean up.
                
                count=1;
                for(String p1:paraSpace1){
                    for(double p2:paraSpace2){
                        for(double p3:paraSpace3){
                            for(boolean p4:paraSpace4){
                                File f= new File(resultsPath+count+".csv");
                                boolean deleted=f.delete();
                                 if(!deleted){
                                     System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                                //                                System.out.println(f.getAbsoluteFile());
                                //                                System.out.println(f.getAbsolutePath());
                                     f.setReadable(true);
                                     f.setWritable(true);
                                     deleted=f.delete();
                                 if(!deleted)
                                     System.out.println("\t DELETE FAILED AGAIN"+resultsPath+count+".csv");

                                 }
                                 count++;
                            }
                        }
                    }
                }
            }else//Not all present, just ditch
                System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
        }
        else{
            ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            bestNumNodes=best.nodes;
            bestLearningRate=best.lRate;
            bestMomentum=best.mRate;
            bestDecay=best.decay;
/* SET UP FINAL MODEL */
            this.setHiddenLayers(bestNumNodes);
            this.setLearningRate(bestLearningRate);
            this.setMomentum(bestMomentum);
            this.setDecay(bestDecay);
            res=best.res;
         }     
    }

   
    @Override
    public void buildClassifier(Instances data) throws Exception{
        
//        res.buildTime=System.currentTimeMillis(); //removed with cv changes  (jamesl) 
        long startTime=System.currentTimeMillis(); 
        //now calced separately from any instance on ClassifierResults, and added on at the end
        int folds=MAX_FOLDS;
        if(folds>data.numInstances())
            folds=data.numInstances();
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        super.setSeed(seed);
        if(tuneParameters){
            if(paraSpace1==null)
                setStandardParaSearchSpace(data.numAttributes()-1);            
            tuneMLP(data);
        }
/*If there is no parameter search, then there is no train CV available.        
this gives the option of finding one. It is inefficient
*/        
        else if(findTrainAcc){
            MultilayerPerceptron t= new MultilayerPerceptron();
            t.setHiddenLayers(this.getHiddenLayers());
            t.setLearningRate(this.getLearningRate());
            t.setMomentum(this.getMomentum());
            t.setDecay(this.getDecay());
            
            CrossValidator cv = new CrossValidator();
            cv.setSeed(seed); 
            cv.setNumFolds(folds);
            cv.buildFolds(data);
            res = cv.crossValidateWithStats(t, data);
        }
//        
        
        super.buildClassifier(data);
        res.buildTime=System.currentTimeMillis()-startTime;
        if(trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(data.relationName()+",TunedMLP,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeString(res.writeInstancePredictions());
        }
    }
   
    public static void main(String[] args) {
//        jamesltests();
        TunedTwoLayerMLP t=new TunedTwoLayerMLP();
        t.debug=true;
        for(int i=1;i<=1024;i++)
            t.setParametersFromIndex(i);
        System.exit(0);
        
        DecimalFormat df = new DecimalFormat("##.###");
        try{
            String dset = "balloons";             
           Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\"+dset+"\\"+dset);        
            Instances[] split=InstanceTools.resampleInstances(all,1,0.5);
                TunedTwoLayerMLP rf=new TunedTwoLayerMLP();
                rf.debug(true);
                rf.tuneParameters(true);
               rf.buildClassifier(split[0]);
         }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
       
    }
   
    public static class ResultsHolder{
        String nodes;
        double lRate,mRate;
        boolean decay;
        ClassifierResults res;
        ResultsHolder(String a, double b,double c, boolean d, ClassifierResults r){
            nodes=a;
            lRate=b;
            mRate=0;
            decay=d;
            res=r;
        }
    }

   
}
