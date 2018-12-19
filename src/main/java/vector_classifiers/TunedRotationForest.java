/*

Rotation forest parameters

feature partition size. This can be randomized, but we just keep it as fixed
t.setMaxGroup(numFeatures);
t.setMinGroup(numFeatures);
Default is 3. Search range of
3,4,...,12 (or m, whichever is bigger)

m_RemovedPercentage. note percentage removed, not kept
default is 50%. Search range is
0,10,20,30,40,50,60,70,80,90

m_NumIterations. Number of iterations.
default is 10. Search range is
100,200,300,400,500,600,700,800,900,1000

PLUS: all the tree parameters
minimum per leaf, pruning, confidence etc.
 */
package vector_classifiers;

import development.CollateResults;
import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import weka.classifiers.trees.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.RotationForest;
import utilities.ClassifierResults;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ajb
 */
public class TunedRotationForest extends RotationForest implements SaveParameterInfo,TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable{
    protected boolean tuneParameters=true;
    protected int[] paraSpace1;//Number of features parameter
    protected int[] paraSpace2;//Percentage to remove
    protected int[] paraSpace3;//Number of trees parameter
    protected int[] paras;
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
//HARD CODED FLAG that allows a build from partials    
    private boolean buildFromPartial=false;
    public TunedRotationForest(){
        super();
        this.setNumIterations(200);
        rng=new Random();
        seed=0;
        accuracy=new ArrayList<>();
        
    }   
//SaveParameterInfo    
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",RemovePercent,"+this.getRemovedPercentage()+",NumFeatures,"+this.getMaxGroup()+",numTrees,"+this.getNumIterations();
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
        tuneParameters=false;
        paras=new int[3];
//Para 1: group size. 
        int numPerGroupIndex=x/(MAX_PER_PARA*MAX_PER_PARA);
        int removePercentIndex=(x%MAX_PER_PARA)/MAX_PER_PARA;
        int numTreesIndex=x%MAX_PER_PARA;
        paras[0]=3+numPerGroupIndex;
        paras[1]=(100/MAX_PER_PARA)*removePercentIndex;
        
        if(numTreesIndex==0)
            paras[2]=10;
        else
            paras[2]=50*numTreesIndex;
//NOTE THE BASE CLASSIFIER CHECKS THAT minGroup and maxGroup do not exceed 
//nos attributes so if the index means they do, they will simply be reset to m, 
 //the number of Attributes
            this.setMaxGroup(paras[0]);
            this.setMinGroup(paras[0]);
            this.setRemovedPercentage(paras[1]);
            this.setNumIterations(paras[2]);
           
        
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
    public void setNumTreesRange(int[] d){
        paraSpace2=d;
    }
    public void setNumFeaturesRange(int[] d){
        paraSpace1=d;
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
        int numParas=MAX_PER_PARA;
        if(m<numParas)
            numParas=m;
        paraSpace1=new int[numParas]; //m_MinGroup and m_MaxGroup
        paraSpace1[0]=3;
        for(int i=1;i<numParas;i++)//Will need to check in build classifier that this does not exceed m
            paraSpace1[i]=3+i;
        paraSpace2=new int[MAX_PER_PARA];//m_RemovedPercentage
        paraSpace2[0]=0;
        for(int i=1;i<MAX_PER_PARA;i++)
        paraSpace2[i]=(100/MAX_PER_PARA)*i;
        paraSpace3=new int[MAX_PER_PARA];//Number of trees
        for(int i=0;i<paraSpace3.length;i++)
            paraSpace3[i]=(50)*(i+1);  
//        System.out.println("Number of parameters for each ="+paraSpace1.length*paraSpace2.length*paraSpace3.length);
            
    }

    public void tuneRotationForest(Instances train) throws Exception {
         paras=new int[3];
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
        ArrayList<TunedSVM.ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(int p1:paraSpace1){//Num atts in group
            for(int p2:paraSpace2){//Percent removed
                for(int p3:paraSpace3){//Num trees
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
                    TunedRotationForest model = new TunedRotationForest();
                    model.tuneParameters(false);
                    model.findTrainAcc=false;
                    model.setMaxGroup(p1);
                    model.setMinGroup(p1);
                    model.setRemovedPercentage(p2);
                    model.setNumIterations(p3);
                    model.paras=new int[3];
                   
                    tempResults=cv.crossValidateWithStats(model,trainCopy);
                    tempResults.setName("RotFPara"+count);
                    tempResults.setParas("NumFeatures,"+p1+",RemovePercent,"+p2+",numTrees,"+p3);
                    double e=1-tempResults.acc;
                    if(debug)
                        System.out.println("Group size="+p1+",Remove Prop="+p2+",Trees="+p3+", Acc = "+(1-e));
                    accuracy.add(tempResults.acc);
                    if(saveEachParaAcc){// Save to file and close
                        temp=new OutFile(resultsPath+count+".csv");
                        temp.writeLine(tempResults.writeResultsFileToString());
                        temp.closeFile();
                        File f=new File(resultsPath+count+".csv");
                        if(f.exists())
                            f.setWritable(true, false);
                         
                    }                
                    else{
                        if(e<minErr){
                        minErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                        }
                        else if(e==minErr){//Sort out ties
                            ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                        }
                    }
                }
            }
        }
//Step 2: set the actual parameters        
        int bestNumAtts;
        int bestRemovePercent;
        int bestNumTrees;
        minErr=1;
        if(saveEachParaAcc){
// Check they are all there first. 
            int missing=0;
            for(int p1:paraSpace1){
                for(int p2:paraSpace2){
                    for(int p3:paraSpace3){
                        File f=new File(resultsPath+count+".csv");
                        if(!(f.exists() && f.length()>0))
                            missing++;
                    }
                }
            }
            if(missing==0)//All present, or at least one and build from partial
            {
                combinedBuildTime=0;
    //            If so, read them all from file, pick the best
                count=0;
                for(int p1:paraSpace1){//num per group
                    for(int p2:paraSpace2){//removal percentage
                        for(int p3:paraSpace3){//num trees
                            count++;
                            tempResults = new ClassifierResults();
                            if(new File(resultsPath+count+".csv").exists()){
                                tempResults.loadFromFile(resultsPath+count+".csv");
                                combinedBuildTime+=tempResults.buildTime;
                                double e=1-tempResults.acc;
                                if(e<minErr){
                                    minErr=e;
                                    ties=new ArrayList<>();//Remove previous ties
                                    ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                                }
                                else if(e==minErr){//Sort out ties
                                        ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                                }
                            }
                        }            
                    }
                }
                TunedSVM.ResultsHolder best=ties.get(rng.nextInt(ties.size()));
                bestNumAtts=(int)best.x;
                bestRemovePercent=(int)best.y;
                bestNumTrees=(int)best.z;
                paras[0]=bestNumAtts;
                paras[1]=bestRemovePercent;
                paras[2]=bestNumTrees;
                this.setNumIterations(bestNumTrees);
                this.setRemovedPercentage(bestRemovePercent);
                this.setMaxGroup(bestNumAtts);
                this.setMinGroup(bestNumAtts);
//                this.setMaxDepth(bestNumLevels);
//                this.setNumFeatures(bestNumFeatures);
               
                res=best.res;
                if(debug)
                    System.out.println("Bestnum in group ="+bestNumAtts+"  best num trees ="+bestNumTrees+" best train acc = "+res.acc);
//File clean up, delete all the individual fold files
        //Delete the files here to clean up.
                
                 count=1;
                for(int p1:paraSpace1){//num per group
                    for(int p2:paraSpace2){//removal percentage
                        for(int p3:paraSpace3){//num trees
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
            }else//Not all present, just ditch
                System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
        }
        else{
            TunedSVM.ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            bestNumAtts=(int)best.x;
            bestRemovePercent=(int)best.y;
            bestNumTrees=(int)best.z;
            paras[0]=bestNumAtts;
            paras[1]=bestRemovePercent;
            paras[2]=bestNumTrees;
            this.setNumIterations(bestNumTrees);
            this.setRemovedPercentage(bestRemovePercent);
            this.setMaxGroup(bestNumAtts);
            this.setMinGroup(bestNumAtts);
            res=best.res;
         }     
    }
    private void setParasFromPartiallyCompleteSearch() throws Exception{
         paras=new int[3];
        combinedBuildTime=0;
        ArrayList<TunedSVM.ResultsHolder> ties=new ArrayList<>();
    //            If so, read them all from file, pick the best
        int count=0;
        int present=0;
        double minErr=1;
        for(int p1:paraSpace1){//num per group
            for(int p2:paraSpace2){//removal percentage
                for(int p3:paraSpace3){//num trees
                    ClassifierResults tempResults = new ClassifierResults();
                    count++;
                    if(new File(resultsPath+count+".csv").exists()){
                        present++;
                        tempResults.loadFromFile(resultsPath+count+".csv");
                        combinedBuildTime+=tempResults.buildTime;
                        double e=1-tempResults.acc;
                        if(e<minErr){
                            minErr=e;
                            ties=new ArrayList<>();//Remove previous ties
                            ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                        }
                        else if(e==minErr){//Sort out ties
                                ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                        }
                    }
                }            
            }
        }
//Set the parameters
        if(present>0){
            System.out.println("Number of paras = "+present);
            System.out.println("Number of best = "+ties.size());
            TunedSVM.ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            int bestNumAtts;
            int bestRemovePercent;
            int bestNumTrees;
            bestNumAtts=(int)best.x;
            bestRemovePercent=(int)best.y;
            bestNumTrees=(int)best.z;
            paras[0]=bestNumAtts;
            paras[1]=bestRemovePercent;
            paras[2]=bestNumTrees;
            this.setNumIterations(bestNumTrees);
            this.setRemovedPercentage(bestRemovePercent);
            this.setMaxGroup(bestNumAtts);
            this.setMinGroup(bestNumAtts);
            res=best.res;
        }        
        else
            throw new Exception("Error, no parameter files for "+resultsPath);
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
        if(buildFromPartial){
            if(paraSpace1==null)
                setStandardParaSearchSpace(data.numAttributes()-1);            
            setParasFromPartiallyCompleteSearch();            
        }
        else if(tuneParameters){
            if(paraSpace1==null)
                setStandardParaSearchSpace(data.numAttributes()-1);            
            tuneRotationForest(data);
        }
/*If there is no parameter search, then there is no train CV available.        
this gives the option of finding one using 10xCV  
*/        
        else if(findTrainAcc){
            RotationForest t= new RotationForest();
            t.setMaxGroup(this.getMaxGroup());
            t.setMinGroup(this.getMinGroup());
            t.setNumIterations(this.getNumIterations());
            t.setSeed(seed);
            
            //new (jamesl) 
            CrossValidator cv = new CrossValidator();
            cv.setSeed(seed); //trying to mimick old seeding behaviour below
            cv.setNumFolds(folds);
            cv.buildFolds(data);
            res = cv.crossValidateWithStats(t, data);
        }
//        
        
        super.buildClassifier(data);
        res.buildTime=System.currentTimeMillis()-startTime;
        if(trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(data.relationName()+",TunedRotF,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeString(res.writeInstancePredictions());
            f.closeFile();
            File x=new File(trainPath);
            x.setWritable(true, false);
             
        }
    }
  
    public static void jamesltests() {
        //tests to confirm correctness of cv changes
        //summary: pre/post change avg accs over 50 folds
        // train: 0.9770149253731346 vs 0.9752238805970153
        // test: 0.9672886297376094 vs 0.9673858114674441
        
        //post change trainaccs/testaccs on 50 folds of italypowerdemand: 
        //      avg trainacc=0.9752238805970153
        //folds: 
//        [0.9850746268656716, 0.9701492537313433, 1.0, 1.0, 0.9552238805970149, 
//        0.9850746268656716, 0.9850746268656716, 0.9701492537313433, 0.9552238805970149, 1.0, 
//        0.9552238805970149, 0.9850746268656716, 0.9701492537313433, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9701492537313433, 1.0, 0.9850746268656716, 
//        0.9850746268656716, 0.9701492537313433, 0.9552238805970149, 0.9850746268656716, 1.0, 
//        0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 
//        0.9701492537313433, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9850746268656716, 0.9701492537313433, 
//        0.9850746268656716, 1.0, 0.9253731343283582, 0.9701492537313433, 0.9850746268656716, 
//        0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9850746268656716]

//              avg testacc=0.9673858114674441
//        folds: 
//        [0.9727891156462585, 0.9484936831875608, 0.966958211856171, 0.9708454810495627, 0.9747327502429544, 
//        0.9718172983479106, 0.9659863945578231, 0.9737609329446064, 0.9504373177842566, 0.9689018464528668, 
//        0.9689018464528668, 0.9708454810495627, 0.9718172983479106, 0.9737609329446064, 0.9698736637512148, 
//        0.9727891156462585, 0.9465500485908649, 0.9727891156462585, 0.9689018464528668, 0.9708454810495627, 
//        0.9718172983479106, 0.9727891156462585, 0.9630709426627794, 0.9718172983479106, 0.967930029154519, 
//        0.9659863945578231, 0.967930029154519, 0.9659863945578231, 0.9494655004859086, 0.9698736637512148, 
//        0.9504373177842566, 0.9718172983479106, 0.9708454810495627, 0.9698736637512148, 0.9659863945578231, 
//        0.9630709426627794, 0.9718172983479106, 0.9698736637512148, 0.9708454810495627, 0.9737609329446064, 
//        0.9689018464528668, 0.9698736637512148, 0.9514091350826045, 0.9698736637512148, 0.9650145772594753, 
//        0.967930029154519, 0.9737609329446064, 0.9650145772594753, 0.9718172983479106, 0.9689018464528668]
        
        //pre change trainaccs/testaccs on 50 folds of italypowerdemand: 
//            avg trainacc=0.9770149253731346
//        folds:
//        [0.9850746268656716, 1.0, 1.0, 1.0, 0.9552238805970149, 
//        0.9850746268656716, 0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 1.0, 
//        0.9552238805970149, 0.9850746268656716, 0.9701492537313433, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9402985074626866, 0.9701492537313433, 1.0, 0.9850746268656716, 
//        0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 1.0, 
//        0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 
//        0.9701492537313433, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9701492537313433, 0.9850746268656716, 0.9701492537313433, 
//        0.9850746268656716, 1.0, 0.9253731343283582, 0.9701492537313433, 0.9850746268656716, 
//        0.9402985074626866, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716]

//                avg testacc=0.9672886297376094
//        folds:
//        [0.9737609329446064, 0.9484936831875608, 0.9689018464528668, 0.9698736637512148, 0.9737609329446064, 
//        0.9708454810495627, 0.9650145772594753, 0.9718172983479106, 0.93488824101069, 0.966958211856171, 
//        0.966958211856171, 0.9698736637512148, 0.9708454810495627, 0.9718172983479106, 0.9698736637512148, 
//        0.9718172983479106, 0.9426627793974732, 0.9727891156462585, 0.967930029154519, 0.9708454810495627, 
//        0.9708454810495627, 0.9727891156462585, 0.9650145772594753, 0.9718172983479106, 0.9698736637512148, 
//        0.9698736637512148, 0.9698736637512148, 0.9689018464528668, 0.9601554907677357, 0.9708454810495627, 
//        0.9582118561710399, 0.9708454810495627, 0.9708454810495627, 0.9708454810495627, 0.9698736637512148, 
//        0.9572400388726919, 0.9727891156462585, 0.9727891156462585, 0.9718172983479106, 0.9737609329446064, 
//        0.9698736637512148, 0.9689018464528668, 0.9329446064139941, 0.9708454810495627, 0.9718172983479106, 
//        0.9708454810495627, 0.9737609329446064, 0.9620991253644315, 0.9718172983479106, 0.9727891156462585]

        
        
        System.out.println("rotftests");
        
        String dataset = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");
        
        int rs = 50;
        
        double[] trainAccs = new double[rs];
        double[] testAccs = new double[rs];
        double trainAcc =0;
        double testAcc =0;
        for (int r = 0; r < rs; r++) {
            Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, r);
            
            TunedRotationForest rotF = new TunedRotationForest();
            rotF.estimateAccFromTrain(true);
            
            try {
                rotF.buildClassifier(data[0]);
            } catch (Exception ex) {
                Logger.getLogger(TunedRotationForest.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            trainAccs[r] = rotF.res.acc;
            trainAcc+=trainAccs[r];
            
            testAccs[r] = ClassifierTools.accuracy(data[1], rotF);
            testAcc+=testAccs[r];
            
            System.out.print(".");
        }
        trainAcc/=rs;
        testAcc/=rs;
        
        System.out.println("\nacc="+trainAcc);
        System.out.println(Arrays.toString(trainAccs));
        
        System.out.println("\nacc="+testAcc);
        System.out.println(Arrays.toString(testAccs));
    }
    
    public static void main(String[] args) {
        cheatOnMNIST();
        System.exit(0);
//        jamesltests();
        DecimalFormat df = new DecimalFormat("##.###");
        try{
            String dset = "balloons";             
           Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\"+dset+"\\"+dset);        
            Instances[] split=InstanceTools.resampleInstances(all,1,0.5);
                TunedRotationForest rf=new TunedRotationForest();
                rf.debug(true);
                rf.tuneParameters(true);
               rf.buildClassifier(split[0]);
         }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
       
    }
    
    public static void cheatOnMNIST(){
        Instances train=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\LargeProblems\\MNIST\\MNIST_TRAIN");
        Instances test=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\LargeProblems\\MNIST\\MNIST_TEST");
        RotationForest rf=new RotationForest();
        System.out.println("Data loaded ......");
        double a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
        System.out.println("Trees ="+10+" acc = "+a);
        for(int trees=50;trees<=500;trees+=50){
            rf.setNumIterations(trees);
            a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
            System.out.println("Trees ="+trees+" acc = "+a);
        }
        
    }
  
}
