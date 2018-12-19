/*
learning rate of 0 meaningless, but 1 fine.  Momentum between 0 and 1 (inclusive) 
is fine.  Decay is probably weight decay regularisation,  difficult to know 
how it is coded and value will depend on the size of the training set and learning rate, I'd try 1, 1/2, 1/4, ... going down to very small values.
MLP parameters searched by this classifier

Single hidden layer version:

number of nodes: 5 values, data dependent: Embedded in m_hiddenLayers. Values either 
"a"=(m_numAttributes + m_numClasses) / 2;
"i"= m_numAttributes;
"o" = m_numClasses;
"t" = m_numAttributes + m_numClasses;
Fixed value =0 to represent no hidden layer
Learning rate: 10 values, data independent, variable m_learningRate
1,1/2,1/4,1/2^9
Momentum: 10 values
0,0.1,0.2,...0.9
m_decay 2 values: either true or false

total models = 5*10*10*2=1000

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
public class TunedSingleLayerMLP extends MultilayerPerceptron implements SaveParameterInfo,TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable{
    protected boolean tuneParameters=true;
    protected String[] paraSpace1;//number of nodes
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
    
    public TunedSingleLayerMLP(){
        super();
        rng=new Random();
        seed=0;
        accuracy=new ArrayList<>();
        
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
//PARA SPACE 5x10x10x2=1000. Going to just hard code the sizes
//        int paraSpaceSize=1000;
        String p1="0";
        if(x<=200)
            p1="0";
        else if(x<=400)
            p1="a";
        else if(x<=600)
            p1="i";
        else if(x<=800)
            p1="o";
        else
            p1="t";
//"a"=(m_numAttributes + m_numClasses) / 2;
//"i"= m_numAttributes;
//"o" = m_numClasses;
//"t" = m_numAttributes + m_numClasses;
        
        int t=(x-1)%200;
        boolean p4;
        if(t<=100)
            p4=false;
        else
            p4=true;
        t=(x-1)%100;
        double p2=Math.pow(2,t%10);
        p2=1.0/p2;
        double p3=t/10;
        p3=p3/10;
 /* SET UP MODEL        */
        this.setHiddenLayers(p1);
        this.setLearningRate(p2);
        this.setMomentum(p3);
        this.setDecay(p4);
        if(debug)
            System.out.println("input ="+x+" Paras ="+p1+","+p2+","+p3+","+p4);
        
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
    protected final void setStandardParaSearchSpace(int m){
        paraSpace1=new String[5]; //# hidden layers
        paraSpace1[0]="0";//No hidden layer
        paraSpace1[1]="a";// =(m_numAttributes + m_numClasses) / 2;
        paraSpace1[2]="i";//=m_numAttributes;
        paraSpace1[3]="o";// = m_numClasses;
        paraSpace1[4]="t";//= m_numAttributes + m_numClasses;    
        paraSpace2=new double[10];//Learning rate
        for(int i=0;i<paraSpace2.length;i++)
            paraSpace2[i]=1.0/Math.pow(2,i);
        paraSpace3=new double[10];//momentum
        for(int i=0;i<paraSpace3.length;i++)
            paraSpace3[i]=i/10.0;
        paraSpace4=new boolean[2];
        paraSpace4[0]=true;
        paraSpace4[1]=false;
        if(debug)
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
                        TunedSingleLayerMLP model = new TunedSingleLayerMLP();
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
        TunedSingleLayerMLP t=new TunedSingleLayerMLP();
        t.debug=true;
        for(int i=1;i<=1000;i++)
            t.setParametersFromIndex(i);
        System.exit(0);
        
        DecimalFormat df = new DecimalFormat("##.###");
        try{
            String dset = "balloons";             
           Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\"+dset+"\\"+dset);        
            Instances[] split=InstanceTools.resampleInstances(all,1,0.5);
                TunedSingleLayerMLP rf=new TunedSingleLayerMLP();
                rf.debug(true);
                rf.tuneParameters(true);
               rf.buildClassifier(split[0]);
         }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
       
    }
   
    static class ResultsHolder{
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
