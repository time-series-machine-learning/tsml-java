/*
Tony's attempt to see the effect of parameter setting on SVM.

Two parameters: 
kernel para: for polynomial this is the weighting given to lower order terms
    k(x,x')=(<x'.x>+b)^d
regularisation parameter, used in the SMO 

m_C
 */
package vector_classifiers;

import development.CollateResults;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
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
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import utilities.ClassifierResults;
import weka.core.*;

/**
 *
 * @author ajb
 
 TunedSVM sets the margin c through b ten fold cross validation.
 
 If the kernel type is RBF, also set sigma through CV, same values as c
 
 NOTE: 
 1. CV could be done faster?
 2. Could use libSVM instead
 * 
 */
public class TunedSVM extends SMO implements SaveParameterInfo, TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable{
    boolean setSeed=false;
    int seed;
    int minC=-16;//These search values are used for all kernels with C. It is also used for Gamma in RBF, but not for the Polynomial exponent search
    int maxC=16;
    int minExponent=1;//These values are also used for Gamma in RBF, but not for the Polynomial exponent search
    int maxExponent=6;
    int minB=0;//These are for the constant value in the Polynomial Kernel
    int maxB=5;
    double IncrementB=1;
    double[] paraSpace1;//For fixed polynomial (LINEAR and QUADRATIC) there is just one range of parameters 
    double[] paraSpace2;//For RBF this is gamma, for POLYNOMIAL it is exponent. 
    double[] paraSpace3;//For POLYNOMIAL this is the constant term b in the kernel.
    private static int MAX_FOLDS=10;
    private double[] paras;//Stored final parameter values after search
    String trainPath="";
    boolean debug=false;
    protected boolean findTrainAcc=true;
    Random rng;
    ArrayList<Double> accuracy;
    private boolean kernelOptimise=false;   //Choose between linear, quadratic and RBF kernel
    private boolean tuneParameters=true;
    private ClassifierResults res =new ClassifierResults();
    private long combinedBuildTime;
    private boolean buildFromFile=false;
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
//HARD CODED FLAG that allows a build from partials    
    private boolean buildFromPartial=false;
    
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }
    public TunedSVM(){
        super();
        kernelOptimise=false;
        kernel=KernelType.RBF;
        tuneParameters=true;
        setKernel(new RBFKernel());
        rng=new Random();
        accuracy=new ArrayList<>();
        setBuildLogisticModels(true);
    }
    public void estimateAccFromTrain(boolean b){
        this.findTrainAcc=b;
    }

    
    public void setSeed(int s){
        this.setSeed=true;
        seed=s;
        rng=new Random();
        rng.setSeed(seed);
    }
    
 @Override
    public void writeCVTrainToFile(String train) {
        findTrainAcc=true;
        trainPath=train;
    }    
 @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        findTrainAcc=setCV;
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
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc;
        result+=",C,"+paras[0];
        if(paras.length>1){
            if(kernel==KernelType.RBF)
                result+=",Gamma,"+paras[1];
            else if (paras.length>2 && kernel==KernelType.POLYNOMIAL)
                result+=",Power,"+paras[1]+",b,"+paras[2];
        }
       for(double d:accuracy)
            result+=","+d;
        
        return result;
    }

    @Override
    public void setParamSearch(boolean b) {
        tuneParameters=b;
    }

    @Override
    public void setParametersFromIndex(int x) {
        kernelOptimise=false;   //Choose between linear, quadratic and RBF kernel
        tuneParameters=false;
        int numCParas=maxC-minC+1;
        
        
        if(kernel==KernelType.LINEAR || kernel==KernelType.QUADRATIC){//Single parameter for C between 1 and 33
            if(x<1 || x>numCParas)//Error, invalid range
                throw new UnsupportedOperationException("ERROR parameter index "+x+" out of range "+minC+" to "+ "max"); //To change body of generated methods, choose Tools | Templates.
            paras=new double[1];
            paras[0]=Math.pow(2,minC+(x-1));
            setC(paras[0]);
        }
        else if(kernel==KernelType.RBF){//Two parameters, same range for both
            if(x<1 || x>numCParas*numCParas)//Error, invalid range
                throw new UnsupportedOperationException("ERROR parameter index "+x+" out of range "+minC+" to "+ "max"); //To change body of generated methods, choose Tools | Templates.
            paras=new double[2];
            int temp=minC+(x-1)/numCParas;
            paras[0]=Math.pow(2,temp);
            temp=minC+(x-1)%numCParas;
            paras[1]=Math.pow(2,temp);
            setC(paras[0]);
            ((RBFKernel)m_kernel).setGamma(paras[1]);
            System.out.println("");
        }
        else if(kernel==KernelType.POLYNOMIAL){
//Three paras, not evenly distributed. C [1  to 33] exponent =[1 to 6], b=[0 to 5] 
            paras=new double[3];
            int numExpParas=maxExponent-minExponent+1;
            int numBParas=maxB-minB+1;
            if(x<1 || x>numCParas*numExpParas*numBParas)//Error, invalid range
                throw new UnsupportedOperationException("ERROR parameter index "+x+" out of range for PolyNomialKernel"); //To change body of generated methods, choose Tools | Templates.
            int cPara=minC+(x-1)%numCParas;
            int expPara=minExponent+(x-1)/(numBParas*numCParas);
            int bPara=minB+((x-1)/numCParas)%numBParas;
            paras[0]=Math.pow(2,cPara);
            paras[1]=expPara;
            paras[2]=bPara;
              PolynomialKernel kern = new PolynomialKernel();
            kern.setExponent(paras[1]);
            kern.setB(paras[2]);
            setKernel(kern);
            setC(paras[0]);
            System.out.println("Index "+x+" maps to "+cPara+","+expPara+","+bPara);
         }
    }

    @Override
    public String getParas() { //This is redundant really.
        return getParameters();
    }

    @Override
    public double getAcc() {
        return res.acc;
    }
    public enum KernelType {LINEAR,QUADRATIC,POLYNOMIAL,RBF};
    KernelType kernel;
    public void debug(boolean b){
        this.debug=b;
    }

    public void setKernelType(KernelType type) {
        kernel = type;
        switch (type) {
            case LINEAR:                     
                PolyKernel p=new PolynomialKernel();
                p.setExponent(1);
                setKernel(p);
            break;
            case QUADRATIC:
                PolyKernel p2=new PolynomialKernel();
                p2.setExponent(2);
                setKernel(p2);
            break;
            case POLYNOMIAL:
                PolyKernel p3=new PolynomialKernel();
                p3.setExponent(1);
                setKernel(p3);
            break;
            case RBF:
                RBFKernel kernel2 = new RBFKernel();
                setKernel(kernel2);
            break;
        }
    }
    
    public void setParaSpace(double[] p){
        paraSpace1=p;
    }
    public void setStandardParaSearchSpace(){
        paraSpace1=new double[maxC-minC+1];
        for(int i=minC;i<=maxC;i++)
            paraSpace1[i-minC]=Math.pow(2,i);
        if(kernel==KernelType.RBF){
            paraSpace2=new double[maxC-minC+1];
            for(int i=minC;i<=maxC;i++)
                paraSpace2[i-minC]=Math.pow(2,i);
        }
        else if(kernel==KernelType.POLYNOMIAL){
            paraSpace2=new double[maxExponent-minExponent+1];
            paraSpace3=new double[maxB-minB+1];
            for(int i=minExponent;i<=maxExponent;i++)
                    paraSpace2[i-minExponent]=i;
            for(int i=minB;i<=maxB;i++)
                    paraSpace3[i-minB]=i;
        }
    }
/**
 * 
 * @param n number of parameter values to try, spread across 2^minC and 2^maxC
 on an exponential scale
 */
    public void setLargePolynomialParameterSpace(int n){
        paraSpace1=new double[n];
        double interval=(maxC-minC)/(n-1);
        double exp=minC;
        for(int i=0;i<n;i++){
          paraSpace1[i]=  Math.pow(2,exp); 
          exp+=interval;
        }
    }

    
    public void optimiseKernel(boolean b){kernelOptimise=b;}
    
    public boolean getOptimiseKernel(){ return kernelOptimise;}
    public void optimiseParas(boolean b){tuneParameters=b;}
    
    static class ResultsHolder{
        double x,y,z;
        ClassifierResults res;
        ResultsHolder(double a, double b,ClassifierResults r){
            x=a;
            y=b;
            z=0;
            res=r;
        }
        ResultsHolder(double a, double b,double c,ClassifierResults r){
            x=a;
            y=b;
            z=c;
            res=r;
        }
    }

    public void tuneRBF(Instances train) throws Exception {
        paras=new double[2];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();

        double minErr=1;
        this.setSeed(rng.nextInt());
        
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        
        
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(double p1:paraSpace1){
            for(double p2:paraSpace2){
                count++;
                if(saveEachParaAcc){// check if para value already done
                    File f=new File(resultsPath+count+".csv");
                    if(f.exists()){
                        if(f.length()==0){//Empty, delete
                            f.delete();
                        }
                        else
                            continue;//If done, ignore skip this iteration                        
                    }
                }
                SMO model = new SMO();
                RBFKernel kern = new RBFKernel();
                kern.setGamma(p2);
                model.setKernel(kern);
                model.setC(p1);
                model.setBuildLogisticModels(true);
                tempResults=cv.crossValidateWithStats(model,trainCopy);
                tempResults.setName("TunedSVM"+kernel);
                tempResults.setParas("C,"+p1+",Gamma,"+p2);

//                Evaluation eval=new Evaluation(temp);
//                eval.crossValidateModel(model, temp, folds, rng);
                double e=1-tempResults.acc;
                accuracy.add(tempResults.acc);
                if(debug)
                    System.out.println(" C= "+p1+" Gamma = "+p2+" Acc = "+(1-e));
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
                    ties.add(new ResultsHolder(p1,p2,tempResults));
                    }
                    else if(e==minErr){//Sort out ties
                        ties.add(new ResultsHolder(p1,p2,tempResults));
                    }
                }
            }
        }
        double bestC;
        double bestSigma;
        minErr=1;
        if(saveEachParaAcc){
// Check they are all there first. 
            int missing=0;
            for(double p1:paraSpace1){
                for(double p2:paraSpace1){
                    File f=new File(resultsPath+count+".csv");
                    if(!(f.exists() && f.length()>0))
                        missing++;
                }
              }
            
            if(missing==0)//All present
            {
                combinedBuildTime=0;
    //            If so, read them all from file, pick the best
                count=0;
                for(double p1:paraSpace1){
                    for(double p2:paraSpace1){
                        count++;
                        tempResults = new ClassifierResults();
                        tempResults.loadFromFile(resultsPath+count+".csv");
                        combinedBuildTime+=tempResults.buildTime;
                        double e=1-tempResults.acc;
                        if(e<minErr){
                            minErr=e;
                            ties=new ArrayList<>();//Remove previous ties
                            ties.add(new ResultsHolder(p1,p2,tempResults));
                        }
                        else if(e==minErr){//Sort out ties
                                ties.add(new ResultsHolder(p1,p2,tempResults));
                        }
    //Delete the files here to clean up.

                        File f= new File(resultsPath+count+".csv");
                        if(!f.delete())
                            System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                    }            
                }
                ResultsHolder best=ties.get(rng.nextInt(ties.size()));
                bestC=best.x;
                bestSigma=best.y;
                paras[0]=bestC;
                setC(bestC);
                ((RBFKernel)m_kernel).setGamma(bestSigma);
                paras[1]=bestSigma;
                res=best.res;
                if(debug)
                    System.out.println("Best C ="+bestC+" best Gamma = "+bestSigma+" best train acc = "+res.acc);
            }else//Not all present, just ditch
                System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
        }
        else{
            ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            bestC=best.x;
            bestSigma=best.y;
            paras[0]=bestC;
            setC(bestC);
            ((RBFKernel)m_kernel).setGamma(bestSigma);
            paras[1]=bestSigma;
            res=best.res;
         }
        
    }
/**
 * Searches the polynomial exponent and the C value
 * @param train
 * @throws Exception 
 */    
    public void tunePolynomial(Instances train) throws Exception {

        paras=new double[3];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();

        double minErr=1;
        this.setSeed(rng.nextInt());
        
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        
        
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(double p1:paraSpace1){//C
            for(double p2:paraSpace2){//Exponent
                for(double p3:paraSpace3){//B
                
                    count++;
                    if(saveEachParaAcc){// check if para value already done
                        File f=new File(resultsPath+count+".csv");
                        if(f.exists()){
                            if(CollateResults.validateSingleFoldFile(resultsPath+count+".csv")==false){
                                System.out.println("Deleting file "+resultsPath+count+".csv because incomplete, size ="+f.length());
                            }
                            else
                                continue;//If done, ignore skip this iteration                        
                        }
                    }
                    SMO model = new SMO();
                    PolynomialKernel kern = new PolynomialKernel();
                    kern.setExponent(p2);
                    kern.setB(p3);
                    model.setKernel(kern);
                    model.setC(p1);
                    model.setBuildLogisticModels(true);
                    tempResults=cv.crossValidateWithStats(model,trainCopy);

    //                Evaluation eval=new Evaluation(temp);
    //                eval.crossValidateModel(model, temp, folds, rng);
                    double e=1-tempResults.acc;
                    accuracy.add(tempResults.acc);
                    if(debug)
                        System.out.println("C="+p1+",Exp="+p2+",B="+p3+", Acc = "+(1-e));
                    if(saveEachParaAcc){// Save to file and close
                        temp=new OutFile(resultsPath+count+".csv");
                        temp.writeLine(tempResults.writeResultsFileToString());
                        temp.closeFile();
                    }                
                    else{
                        if(e<minErr){
                        minErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new ResultsHolder(p1,p2,p3,tempResults));
                        }
                        else if(e==minErr){//Sort out ties
                            ties.add(new ResultsHolder(p1,p2,p3,tempResults));
                        }
                    }
                }
            }
        }
        double bestC;
        double bestExponent;
        double bestB;
        
        minErr=1;
        if(saveEachParaAcc){
// Check they are all there first. 
            int missing=0;
            for(double p1:paraSpace1){
                for(double p2:paraSpace2){
                    for(double p3:paraSpace3){
                        File f=new File(resultsPath+count+".csv");
                        if(!(f.exists() && f.length()>0))
                            missing++;
                    }
                }
              }
            
            if(missing==0)//All present
            {
                combinedBuildTime=0;
    //            If so, read them all from file, pick the best
                count=0;
                for(double p1:paraSpace1){//C
                    for(double p2:paraSpace2){//Exponent
                        for(double p3:paraSpace3){//B
                            count++;
                            tempResults = new ClassifierResults();
                            tempResults.loadFromFile(resultsPath+count+".csv");
                            combinedBuildTime+=tempResults.buildTime;
                            double e=1-tempResults.acc;
                            if(e<minErr){
                                minErr=e;
                                ties=new ArrayList<>();//Remove previous ties
                                ties.add(new ResultsHolder(p1,p2,p3,tempResults));
                            }
                            else if(e==minErr){//Sort out ties
                                    ties.add(new ResultsHolder(p1,p2,p3,tempResults));
                            }
        //Delete the files here to clean up.

                            File f= new File(resultsPath+count+".csv");
                            if(!f.delete())
                                System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                        }
                    }            
                }
                ResultsHolder best=ties.get(rng.nextInt(ties.size()));
                bestC=best.x;
                bestExponent=best.y;
                bestB=best.z;
                paras[0]=bestC;
                paras[1]=bestExponent;
                paras[2]=bestB;
                PolynomialKernel kern = new PolynomialKernel();
                kern.setExponent(bestExponent);
                kern.setB(bestB);
                setKernel(kern);
                setC(bestC);
                
                res=best.res;
                if(debug)
                    System.out.println("Best C ="+bestC+" best Gamma = "+bestExponent+" best train acc = "+res.acc);
            }else//Not all present, just ditch
                System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
        }
        else{
            ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            bestC=best.x;
            bestExponent=best.y;
            bestB=best.z;
            paras[0]=bestC;
            paras[1]=bestExponent;
            paras[2]=bestB;
            PolynomialKernel kern = new PolynomialKernel();
            kern.setExponent(bestExponent);
            kern.setB(bestB);
            setKernel(kern);
            setC(bestC);
            res=best.res;
         }
    }    
    
/**
 * This function assumes the Polynomial exponent is fixed and just searches 
 * for C values. I could generalise this to use with the exponent search, but
 * the risk of introducing bugs is too large
 * @param train
 * @throws Exception 
 */    
   public void tuneCForFixedPolynomial(Instances train) throws Exception {
        paras=new double[1];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();
        double minErr=1;
        this.setSeed(rng.nextInt());
        
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        
        
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;            
        int count=0;
        OutFile temp=null;
        
        for(double d: paraSpace1){
            count++;
            if(saveEachParaAcc){// check if para value already done
                File f=new File(resultsPath+count+".csv");
                if(f.exists() && f.length()>0)
                    continue;//If done, ignore skip this iteration
               if(debug)
                   System.out.println("PARA COUNT ="+count);
            }
            
            SMO model = new SMO();
            model.setKernel(m_kernel);
            model.setC(d);
            model.setBuildLogisticModels(true);
            
            tempResults=cv.crossValidateWithStats(model,trainCopy);
//                Evaluation eval=new Evaluation(temp);
//                eval.crossValidateModel(model, temp, folds, rng);
            double e=1-tempResults.acc;
            accuracy.add(tempResults.acc);
            if(saveEachParaAcc){// Save to file and close
                temp=new OutFile(resultsPath+count+".csv");
                temp.writeLine(tempResults.writeResultsFileToString());
                temp.closeFile();
            }                
            if(e<minErr){
                minErr=e;
               ties=new ArrayList<>();//Remove previous ties
                ties.add(new ResultsHolder(d,0.0,tempResults));
            }
            else if(e==minErr){//Sort out ties
                ties.add(new ResultsHolder(d,0.0,tempResults));
            }
        }
        if(saveEachParaAcc){// Read them all from file, if all donepick the best
            int missing=0;
            for(double p1:paraSpace1){
                File f=new File(resultsPath+count+".csv");
                if(!(f.exists() && f.length()>0))
                    missing++;
            }
            if(missing==0)//All present
            {
                combinedBuildTime=0;
                count=0;
                for(double p1:paraSpace1){
                    count++;
                    tempResults = new ClassifierResults();
                    tempResults.loadFromFile(resultsPath+count+".csv");
                    combinedBuildTime+=tempResults.buildTime;
                    double e=1-tempResults.acc;
                    if(e<minErr){
                        minErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new ResultsHolder(p1,0.0,tempResults));
                    }
                    else if(e==minErr){//Sort out ties
                            ties.add(new ResultsHolder(p1,0.0,tempResults));
                    }
    //Delete the files here to clean up.
                    File f= new File(resultsPath+count+".csv");
                    if(!f.delete())
                        System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                }  
                ResultsHolder best=ties.get(rng.nextInt(ties.size()));
                setC(best.x);
                res=best.res;
                paras[0]=best.x;
            }
            else{
                 System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
            }
        }
        else{
            ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            setC(best.x);
            res=best.res;
            paras[0]=best.x;
        }
    }
     
    public void selectKernel(Instances train) throws Exception {
        KernelType[] ker=KernelType.values();
        double[] rbfParas=new double[2];
        double rbfCVAcc=0;
        double linearBestC=0;
        double linearCVAcc=0;
        double quadraticBestC=0;
        double quadraticCVAcc=0;
        for(KernelType k:ker){
            TunedSVM temp=new TunedSVM();
            Kernel kernel;
            switch(k){
                case LINEAR:                     
                    PolyKernel p=new PolyKernel();
                    p.setExponent(1);
                    temp.setKernel(p);
                    temp.setStandardParaSearchSpace();
                    temp.tuneCForFixedPolynomial(train);
                    linearCVAcc=temp.res.acc;
                    linearBestC=temp.getC();
                break;
                case QUADRATIC:
                    PolyKernel p2=new PolyKernel();
                    p2.setExponent(2);
                    temp.setKernel(p2);
                    temp.setStandardParaSearchSpace();
                    temp.tuneCForFixedPolynomial(train);
                    quadraticCVAcc=temp.res.acc;
                    quadraticBestC=temp.getC();
                break;
                case RBF:
                    RBFKernel kernel2 = new RBFKernel();
                    temp.setKernel(kernel2);
                    temp.setStandardParaSearchSpace();
                    temp.tuneRBF(train);
                    rbfCVAcc=temp.res.acc;
                    rbfParas[0]=temp.getC();
                    rbfParas[1]=((RBFKernel)temp.m_kernel).getGamma();
                    break;
            }
        }
//Choose best, inelligantly
        if(linearCVAcc> rbfCVAcc && linearCVAcc> quadraticCVAcc){//Linear best
            PolyKernel p=new PolyKernel();
            p.setExponent(1);
            setKernel(p);
            setC(linearBestC);
            paras=new double[1];
            paras[0]=linearBestC;
            res.acc=linearCVAcc;
        }else if(quadraticCVAcc> linearCVAcc && quadraticCVAcc> rbfCVAcc){ //Quad best
            PolyKernel p=new PolyKernel();
            p.setExponent(2);
            setKernel(p);
            setC(quadraticBestC);
            paras=new double[1];
            paras[0]=quadraticBestC;
            res.acc=quadraticCVAcc;
        }else{   //RBF
            RBFKernel kernel = new RBFKernel();
            kernel.setGamma(rbfParas[1]);
            setKernel(kernel);
            setC(rbfParas[0]);
            paras=rbfParas;
            res.acc=rbfCVAcc;
        }
    }

//TO DO: add the option to build from an incomplete parameter set, 
//    without deleting
    public void buildFromFile() throws FileNotFoundException{
        combinedBuildTime=0;
        int count=0;
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;            
        double minErr=1;
        if(kernel==KernelType.LINEAR || kernel==KernelType.QUADRATIC){
            for(double p1:paraSpace1){
                count++;
                tempResults = new ClassifierResults();
                File f= new File(resultsPath+count+".csv");
                if(f.exists() && f.length()>0){
                    tempResults.loadFromFile(resultsPath+count+".csv");
                    combinedBuildTime+=tempResults.buildTime;
                    double e=1-tempResults.acc;
                    if(e<minErr){
                        minErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new ResultsHolder(p1,0.0,tempResults));
                    }
                    else if(e==minErr){//Sort out ties
                            ties.add(new ResultsHolder(p1,0.0,tempResults));
                    }
                }
            }  
        }else if(kernel==KernelType.RBF){
            for(double p1:paraSpace1){
                for(double p2:paraSpace2){
                    count++;
                    tempResults = new ClassifierResults();
                    File f= new File(resultsPath+count+".csv");
                    if(f.exists() && f.length()>0){
                        tempResults.loadFromFile(resultsPath+count+".csv");
                        combinedBuildTime+=tempResults.buildTime;
                        double e=1-tempResults.acc;
                        if(e<minErr){
                            minErr=e;
                            ties=new ArrayList<>();//Remove previous ties
                            ties.add(new ResultsHolder(p1,p2,tempResults));
                        }
                        else if(e==minErr){//Sort out ties
                                ties.add(new ResultsHolder(p1,p2,tempResults));
                        }
                    }
                }  
            }            
        }else if(kernel==KernelType.POLYNOMIAL){
             for(double p1:paraSpace1){
                for(double p2:paraSpace2){
                    for(double p3:paraSpace3){
                        count++;
                        tempResults = new ClassifierResults();
                        File f= new File(resultsPath+count+".csv");
                        if(f.exists() && f.length()>0){
                            tempResults.loadFromFile(resultsPath+count+".csv");
                            combinedBuildTime+=tempResults.buildTime;
                            double e=1-tempResults.acc;
                            if(e<minErr){
                                minErr=e;
                                ties=new ArrayList<>();//Remove previous ties
                                ties.add(new ResultsHolder(p1,p2,p3,tempResults));
                            }
                            else if(e==minErr){//Sort out ties
                                    ties.add(new ResultsHolder(p1,p2,p3,tempResults));
                            }
                        }
                    }
                }  
             } 
        }
        ResultsHolder best=ties.get(rng.nextInt(ties.size()));
        setC(best.x);
        res=best.res;
        paras[0]=best.x;
        if(kernel==KernelType.RBF){
            paras[1]=best.y;
//Set Gamma            
        }else if(kernel==KernelType.POLYNOMIAL){
            paras[1]=best.y;
            paras[2]=best.z;           
        }
        
    }
    private void setRBFParasFromPartiallyCompleteSearch() throws Exception{
         paras=new double[2];
        combinedBuildTime=0;
        ArrayList<TunedSVM.ResultsHolder> ties=new ArrayList<>();
    //            If so, read them all from file, pick the best
        int count=0;
        int present=0;
        double minErr=1;
        for(double p1:paraSpace1){//C
            for(double p2:paraSpace2){//GAMMA
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
                        ties.add(new TunedSVM.ResultsHolder(p1,p2,tempResults));
                    }
                    else if(e==minErr){//Sort out ties
                            ties.add(new TunedSVM.ResultsHolder(p1,p2,tempResults));
                    }
                }
            }
        }
//Set the parameters
        if(present>0){
            System.out.println("Number of paras = "+present);
            System.out.println("Number of best = "+ties.size());
            TunedSVM.ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            double bestC;
            double bestSigma;
            bestC=best.x;
            bestSigma=best.y;
            paras[0]=bestC;
            paras[1]=bestSigma;
            setC(bestC);
            ((RBFKernel)m_kernel).setGamma(bestSigma);
            res=best.res;
        }        
        else
            throw new Exception("Error, no parameter files for "+resultsPath);
    }
    private void setPolynomialParasFromPartiallyCompleteSearch() throws Exception{
         paras=new double[3];
        combinedBuildTime=0;
        ArrayList<TunedSVM.ResultsHolder> ties=new ArrayList<>();
    //            If so, read them all from file, pick the best
        int count=0;
        int present=0;
        double minErr=1;
        for(double p1:paraSpace1){//
            for(double p2:paraSpace2){//
                for(double p3:paraSpace3){//
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
            double bestC;
            double bestB;
            bestC=best.x;
            bestB=best.y;
            paras[0]=bestC;
            paras[1]=bestB;
            setC(bestC);
            ((PolynomialKernel)m_kernel).setB(bestB);
            res=best.res;
        }        
        else
            throw new Exception("Error, no parameter files for "+resultsPath);
    }
 
    
    @Override
    public void buildClassifier(Instances train) throws Exception {
        res =new ClassifierResults();
        long t=System.currentTimeMillis();
//        if(kernelOptimise)
//            selectKernel(train);
        if(buildFromPartial){
            if(paraSpace1==null)
                setStandardParaSearchSpace();  
            if(kernel==KernelType.RBF)
                setRBFParasFromPartiallyCompleteSearch();            
//            else if(kernel==KernelType.LINEAR || kernel==KernelType.QUADRATIC)
//                setFixedPolynomialParasFromPartiallyCompleteSearch();            
                else if(kernel==KernelType.POLYNOMIAL)
                    setPolynomialParasFromPartiallyCompleteSearch();            
        }
        else if(tuneParameters){
            if(paraSpace1==null)
                setStandardParaSearchSpace();
            if(buildFromFile){
                throw new Exception("Build from file in TunedSVM Not implemented yet");
            }else{
                if(kernel==KernelType.RBF)
                    tuneRBF(train); //Tunes two parameters
                else if(kernel==KernelType.LINEAR || kernel==KernelType.QUADRATIC)
                    tuneCForFixedPolynomial(train);//Tunes one parameter
                else if(kernel==KernelType.POLYNOMIAL)
                    tunePolynomial(train);
            }
        }
/*If there is no parameter search, then there is no train CV available.        
this gives the option of finding one using 10xCV  
*/        
        else if(findTrainAcc){
            int folds=10;
            if(folds>train.numInstances())
                folds=train.numInstances();
            SMO model = new SMO();
            model.setKernel(this.m_kernel);
            model.setC(this.getC());
            model.setBuildLogisticModels(true);
            model.setRandomSeed(seed);
            CrossValidator cv = new CrossValidator();
            cv.setSeed(seed); //trying to mimick old seeding behaviour below
            cv.setNumFolds(folds);
            cv.buildFolds(train);
            res = cv.crossValidateWithStats(model, train);
      }        
        
        
        
//If both kernelOptimise and tuneParameters are false, it just builds and SVM        
//With whatever the parameters are set to        
        super.buildClassifier(train);
        
        if(saveEachParaAcc)
            res.buildTime=combinedBuildTime;
        else
            res.buildTime=System.currentTimeMillis()-t;
        if(trainPath!=null && trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(train.relationName()+",TunedSVM"+kernel+",Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeLine(res.writeInstancePredictions());
            f.closeFile();
            File x=new File(trainPath);
            x.setWritable(true, false);
            
        }        
    }
    
    public static void jamesltest() {
        try{ 

            String dset = "zoo";
//            int fold = 0;
            Instances all=ClassifierTools.loadData("C:/UCI Problems/"+dset+"/"+dset);
            
            for (int fold = 0; fold < 30; fold++) {
                
            
                Instances[] split=InstanceTools.resampleInstances(all,fold,0.5);
                Instances train=split[0];
                Instances test=split[1];

                TunedSVM svml = new TunedSVM();
                svml.optimiseParas(true);
                svml.optimiseKernel(false);
                svml.setBuildLogisticModels(true);
                svml.setSeed(fold);                
                svml.setKernelType(TunedSVM.KernelType.LINEAR);
    //
    //            TunedSVM svmq = new TunedSVM();
    //            svmq.optimiseParas(true);
    //            svmq.optimiseKernel(false);
    //            svmq.setBuildLogisticModels(true);
    //            svmq.setSeed(fold);                
    //            svmq.setKernelType(TunedSVM.KernelType.QUADRATIC);
    //
    //            TunedSVM svmrbf = new TunedSVM();
    //            svmrbf.optimiseParas(true);
    //            svmrbf.optimiseKernel(false);
    //            svmrbf.setBuildLogisticModels(true);
    //            svmrbf.setSeed(fold);                
    //            svmrbf.setKernelType(TunedSVM.KernelType.RBF);

                System.out.println("\n\nTSVM_L:");
                svml.buildClassifier(train);
                System.out.println("C ="+svml.getC());
                System.out.println("Train: " + svml.res.acc + " " + svml.res.stddev);
                double accL=ClassifierTools.accuracy(test, svml);
                System.out.println("Test: " + accL);
    //
    //
    //            System.out.println("\n\nTSVM_Q:");
    //            svmq.buildClassifier(train);
    //            System.out.println("C ="+svmq.getC());
    //            System.out.println("Train: " + svmq.res.acc + " " + svmq.res.stddev);
    //            double accQ=ClassifierTools.accuracy(test, svmq);
    //            System.out.println("Test: " + accQ);
    //
    //            System.out.println("\n\nTSVM_RBF:");
    //            svmrbf.buildClassifier(train);
    //            System.out.println("C ="+svmrbf.getC());
    //            System.out.println("Train: " + svmrbf.res.acc + " " + svmrbf.res.stddev);
    //            double accRBF=ClassifierTools.accuracy(test, svmrbf);
    //            System.out.println("Test: " + accRBF);
            }
        }catch(Exception e){
            System.out.println("ffsjava");
            System.out.println(e);
            e.printStackTrace();
        }
    }
    
    public static void testKernel() throws Exception{
        TunedSVM svm= new TunedSVM();
        svm.setKernelType(KernelType.POLYNOMIAL);
        svm.setParamSearch(false);
        svm.setBuildLogisticModels(true);

        String dset = "balloons";             
        svm.setSeed(0);
       Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\"+dset+"\\"+dset);        
        Instances[] split=InstanceTools.resampleInstances(all,1,0.5);
        svm.buildClassifier(split[0]);
    }
    
    
    public static void main(String[] args){
        cheatOnMNIST();
        System.exit(0);
        try {
            testKernel();
        } catch (Exception ex) {
            Logger.getLogger(TunedSVM.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.exit(0);
        int min=-16, max=16;
        int numParas=max-min;
        if(max*min<0)
            numParas++;
        
        for(int x=1;x<=1089;x++){
                    int temp=min+(x-1)/numParas;
            double c=Math.pow(2,temp);
            int temp2=min+(x-1)%numParas;
            double gamma=Math.pow(2,temp2);
            System.out.println("c count ="+temp+" gamma count = "+ temp2+" c="+c+"  gamma ="+gamma);
        }

       System.exit(0);
        
        
//        jamesltest();
        
        
        String sourcePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
        String problemFile="ItalyPowerDemand";
        DecimalFormat df = new DecimalFormat("###.###");
        Instances all=ClassifierTools.loadData(sourcePath+problemFile+"/"+problemFile+"_TRAIN");
        Instances[] split=InstanceTools.resampleInstances(all,0,0.5);
        Instances train=split[0];
        Instances test=split[1];
        try{
            TunedSVM svml=new TunedSVM();
                svml.setPathToSaveParameters("C:\\Temp\\fold1_");
                svml.optimiseParas(true);
                svml.optimiseKernel(false);
                svml.setBuildLogisticModels(true);
                svml.setSeed(0);                
                svml.setKernelType(TunedSVM.KernelType.RBF);
                svml.debug=true;
/*            TunedSVM svmq=new TunedSVM();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            svmq.setKernel(kernel);
            TunedSVM svmrbf=new TunedSVM();
            RBFKernel kernel2 = new RBFKernel();
            kernel2.setGamma(1/(double)(all.numAttributes()-1));
            svmrbf.setKernel(kernel2);
            svmq.buildClassifier(train);
            System.out.println("BUILT QUAD");
            System.out.println(" Optimal C ="+svmq.getC());
           svmrbf.buildClassifier(train);
            System.out.println("BUILT RBF");
            System.out.println(" Optimal C ="+svmrbf.getC());
            double accL=0,accQ=0,accRBF=0;
           accQ=ClassifierTools.accuracy(test, svmq);
           accRBF=ClassifierTools.accuracy(test,svmrbf);

        
*/        
           svml.buildClassifier(train);
            System.out.println("BUILT LINEAR = "+svml);
            System.out.println(" Optimal C ="+svml.getC());
             
            double accL=ClassifierTools.accuracy(test, svml);

            System.out.println("ACC on "+problemFile+": Linear = "+df.format(accL)); //+", Quadratic = "+df.format(accQ)+", RBF = "+df.format(accRBF));
                
         }catch(Exception e){
            System.out.println(" Exception building a classifier = "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }

    
    protected static class PolynomialKernel extends PolyKernel {
//Constant parameter to allow for (x.x+b)^m_exponent. The reason this wraps the 
//Weka kernel is I dont think it possible to include this parameter in Weka        
        double b=0; 
        public void setB(double x){b=x;}
        protected void setConstantTerm(double x){ b=x;}
        @Override   
        protected double evaluate(int id1, int id2, Instance inst1)
            throws Exception {
            double result;
            if (id1 == id2) {
              result = dotProd(inst1, inst1);
            } else {
              result = dotProd(inst1, m_data.instance(id2));
            }
    //    // Replacing this
    //    if (m_lowerOrder) {
    //      result += 1.0;
    //    }            
    //Only change from base class to allow for b constant term, rather than 0/1       
            result += b;

            if (m_exponent != 1.0) {
              result = Math.pow(result, m_exponent);
            }
            return result;
          }
    }
   
    public static void cheatOnMNIST(){
        Instances train=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\LargeProblems\\MNIST\\MNIST_TRAIN");
        Instances test=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\LargeProblems\\MNIST\\MNIST_TEST");
        SMO svm=new SMO();
        RBFKernel k=new RBFKernel();
        svm.setKernel(k);
        System.out.println("Data loaded ......");
        double a =ClassifierTools.singleTrainTestSplitAccuracy(svm, train, test);
        System.out.println("Default acc = "+a);
        int min=1;//These search values are used for all kernels with C. It is also used for Gamma in RBF, but not for the Polynomial exponent search
        int max=6;
        for(double c=min;c<=max;c++)
            for(double r=min;r<=max;r++){
                     svm.setC(Math.pow(2, c));
                     k.setGamma(Math.pow(2, r));
                     svm.setKernel(k);//Just in case ...
                    a =ClassifierTools.singleTrainTestSplitAccuracy(svm, train, test);
                    System.out.println("logC ="+c+" logGamma = "+r+" acc = "+a);
                }
        
    }
  
    
}
